import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from .config import config
from .utils import calculate_bleu


def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(input_shape=(*config['image_size'], 3),
                                             include_top=False, weights="imagenet")

    # freeze feature extractor
    base_model.trainable = False
    base_model_out = base_model.output
    base_model_out = layers.Reshape((-1, 1280))(base_model_out)
    cnn_model = keras.models.Model(base_model.input, base_model_out)
    return cnn_model


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = layers.Dense(embed_dim, activation="relu")
        self.layernorm_1 = layers.LayerNormalization()

    def call(self, inputs, training, mask=None):
        inputs = self.dense_proj(inputs)
        attention_output = self.attention(query=inputs, value=inputs,
                                          key=inputs, attention_mask=None)
        proj_input = self.layernorm_1(inputs + attention_output)
        return proj_input


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=vocab_size,
                                                 output_dim=embed_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length,
                                                    output_dim=embed_dim)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)


class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads, seq_len, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=embed_dim)
        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=embed_dim)
        self.dense_proj = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.embedding = PositionalEmbedding(embed_dim=self.embed_dim,
                                             sequence_length=self.seq_len,
                                             vocab_size=self.vocab_size)

        self.out = layers.Dense(self.vocab_size)
        self.dropout_1 = layers.Dropout(0.1)
        self.dropout_2 = layers.Dropout(0.5)
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, training, mask=None):
        inputs = self.embedding(inputs)
        causal_mask = self.get_causal_attention_mask(inputs)
        inputs = self.dropout_1(inputs, training=training)

        if mask is not None:
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        attention_output_1 = self.attention_1(query=inputs, value=inputs,
                                              key=inputs, attention_mask=combined_mask)

        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(query=out_1, value=encoder_outputs,
                                              key=encoder_outputs, attention_mask=padding_mask)

        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        proj_out = self.layernorm_3(out_2 + proj_output)
        proj_out = self.dropout_2(proj_out, training=training)
        preds = self.out(proj_out)
        return preds

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], axis=0)

        return tf.tile(mask, mult)


class ImageCaptioningModel(keras.Model):
    def __init__(self, cnn_model,
                 encoder, decoder,
                 num_captions_per_image=1):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="accuracy")
        self.bleu_tracker = keras.metrics.Mean(name="bleu")
        self.num_captions_per_image = num_captions_per_image

    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    @staticmethod
    def calculate_accuracy(y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)

    @staticmethod
    def calculate_bleu(y_true, y_pred):
        bleu = calculate_bleu(y_true, y_pred)
        bleu = tf.cast(bleu, dtype=tf.float32)
        return bleu

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        with tf.GradientTape() as tape:
            # 3. Pass image embeddings to encoder

            x = self.cnn_model(batch_img)
            encoder_out = self.encoder(x, training=True)

            batch_seq_inp = batch_seq[:, 0, :-1]
            batch_seq_true = batch_seq[:, 0, 1:]

            # 4. Compute the mask for the input sequence
            mask = tf.math.not_equal(batch_seq_inp, 0)

            # 5. Pass the encoder outputs, sequence inputs along with
            # mask to the decoder
            batch_seq_pred = self.decoder(batch_seq_inp, encoder_out,
                                          training=True, mask=mask)

            # 6. Calculate loss and accuracy
            loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
            acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
            bleu = self.calculate_bleu(batch_seq_true, batch_seq_pred)

            # 7. Update the batch loss and batch accuracy
            batch_loss += loss
            batch_acc += acc

        # 8. Get the list of all the trainable weights
        train_vars = (self.encoder.trainable_variables + self.decoder.trainable_variables)

        # 9. Get the gradients
        grads = tape.gradient(loss, train_vars)

        # 10. Update the trainable weights
        self.optimizer.apply_gradients(zip(grads, train_vars))

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.bleu_tracker.update_state(bleu)
        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result(),
                'bleu': self.bleu_tracker.result()
                }

    def test_step(self, batch_data):
        batch_img, batch_seq = batch_data

        # 1. Get image embeddings

        # 3. Pass image embeddings to encoder

        x = self.cnn_model(batch_img)
        encoder_out = self.encoder(x, training=True)

        batch_seq_inp = batch_seq[:, 0, :-1]
        batch_seq_true = batch_seq[:, 0, 1:]

        # 4. Compute the mask for the input sequence
        mask = tf.math.not_equal(batch_seq_inp, 0)

        # 5. Pass the encoder outputs, sequence inputs along with
        # mask to the decoder
        batch_seq_pred = self.decoder(batch_seq_inp, encoder_out,
                                      training=True, mask=mask)

        # 6. Calculate loss and accuracy
        loss = self.calculate_loss(batch_seq_true, batch_seq_pred, mask)
        acc = self.calculate_accuracy(batch_seq_true, batch_seq_pred, mask)
        bleu = self.calculate_bleu(batch_seq_true, batch_seq_pred)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        self.bleu_tracker.update_state(bleu)
        return {"loss": self.loss_tracker.result(),
                "acc": self.acc_tracker.result(),
                'bleu': self.bleu_tracker.result()
                }

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.acc_tracker]
