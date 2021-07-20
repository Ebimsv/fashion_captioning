import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from utils import calculate_bleu
import re
import numpy as np
import tensorflow as tf
import adabelief_tf
from tensorflow.keras import mixed_precision
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

seed = 111
np.random.seed(seed)
tf.random.set_seed(seed)
# mixed_precision.set_global_policy('mixed_float16')
tf.keras.backend.clear_session()

IMAGES_PATH = "./images/"

IMAGE_SIZE = (299, 299)
VOCAB_SIZE = 10000
SEQ_LENGTH = 14  # max=14

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Number of self-attention heads
NUM_HEADS = 4

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 32
EPOCHS = 8
AUTOTUNE = tf.data.experimental.AUTOTUNE

""" Preparing the dataset"""


def load_captions_data(filename):
    with open(filename) as caption_file:
        caption_mapping = {}
        text_data = []
        caption_data = caption_file.readlines()

        for line in caption_data:
            line = line.rstrip("\n")
            img_name, caption = line.split(",")
            img_name = os.path.join(IMAGES_PATH, img_name.strip())

            if img_name.endswith("jpg"):
                caption = "<start> " + caption.strip() + " <end>"
                text_data.append(caption)

                if img_name in caption_mapping:
                    caption_mapping[img_name].append(caption)
                else:
                    caption_mapping[img_name] = [caption]

        return caption_mapping, text_data


def train_val_split(caption_data, train_size=0.9, shuffle=True):
    # 1. list of all image names
    all_images = list(caption_data.keys())

    if shuffle:
        np.random.shuffle(all_images)

    train_size = int(len(caption_data) * train_size)
    training_data = {img_name: caption_data[img_name] for img_name in all_images[:train_size]}
    validation_data = {img_name: caption_data[img_name] for img_name in all_images[train_size:]}
    return training_data, validation_data


captions_mapping, text_data = load_captions_data("./annotations/train_title.txt")
train_data, valid_data = train_val_split(captions_mapping)

print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")

vectorization = TextVectorization(max_tokens=VOCAB_SIZE, output_mode="int",
                                  output_sequence_length=SEQ_LENGTH,
                                  standardize=custom_standardization)

vectorization.adapt(text_data)


def read_image(img_path, size=IMAGE_SIZE):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, size)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def make_dataset(images, captions):
    img_dataset = tf.data.Dataset.from_tensor_slices(images).map(read_image,
                                                                 num_parallel_calls=AUTOTUNE)

    cap_dataset = tf.data.Dataset.from_tensor_slices(captions).map(vectorization,
                                                                   num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.zip((img_dataset, cap_dataset))
    dataset = dataset.batch(BATCH_SIZE).shuffle(256).prefetch(AUTOTUNE)
    return dataset


def get_cnn_model():
    base_model = efficientnet.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3),
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
    def __init__(self, embed_dim, ff_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=embed_dim)

        self.attention_2 = layers.MultiHeadAttention(num_heads=num_heads,
                                                     key_dim=embed_dim)

        self.dense_proj = keras.Sequential([layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)])

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embedding = PositionalEmbedding(embed_dim=EMBED_DIM,
                                             sequence_length=SEQ_LENGTH,
                                             vocab_size=VOCAB_SIZE)

        self.out = layers.Dense(VOCAB_SIZE)
        self.dropout_1 = layers.Dropout(0.1)
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
    def __init__(self, num_captions_per_image=1, training=True):
        super().__init__()
        self.training = training
        self.cnn_model = efficientnet.EfficientNetB0(input_shape=(*IMAGE_SIZE, 3),
                                                     include_top=False, weights="imagenet")
        self.reshape = layers.Reshape((-1, 1280))

        self.encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=NUM_HEADS)
        self.decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=NUM_HEADS)
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

    def _compute_loss_and_acc(self, batch_data, training=True):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        # 1. Get image embeddings
        for i in range(self.num_captions_per_image):
            with tf.GradientTape() as tape:
                # 3. Pass image embeddings to encoder

                x = self.cnn_model(batch_img)
                x = self.reshape(x)
                encoder_out = self.encoder(x, training=self.training)

                batch_seq_inp = batch_seq[:, i, :-1]
                batch_seq_true = batch_seq[:, i, 1:]

                # 4. Compute the mask for the input sequence
                mask = tf.math.not_equal(batch_seq_inp, 0)

                # 5. Pass the encoder outputs, sequence inputs along with
                # mask to the decoder
                batch_seq_pred = self.decoder(batch_seq_inp, encoder_out,
                                              training=training, mask=mask)

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

        return batch_loss, batch_acc / float(self.num_captions_per_image), bleu

    def train_step(self, batch_data):
        batch_img, batch_seq = batch_data
        batch_loss = 0
        batch_acc = 0

        with tf.GradientTape() as tape:
            # 3. Pass image embeddings to encoder

            x = self.cnn_model(batch_img)
            x = self.reshape(x)
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
        x = self.reshape(x)
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
        return [self.loss_tracker, self.acc_tracker, self.bleu_tracker]


def main(epochs=10):
    train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))
    valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))

    caption_model = ImageCaptioningModel(training=True)
    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1
    valid_images = list(valid_data.keys())
    reference = list(valid_data.values())

    cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    early_stopping = keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/title/old/4_heads', histogram_freq=1)
    checkpoint_filepath = './checkpoints/title'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_bleu',
        mode='max',
        save_best_only=True,
        verbose=0)

    op = adabelief_tf.AdaBeliefOptimizer(learning_rate=0.001,
                                         print_change_log=False)
    caption_model.compile(
        optimizer=op,
        loss=cross_entropy,
    )

    def lr_scheduler(epoch, lr):
        if epoch == 40 or epoch == 80:
            lr = lr / 10
            return lr
        else:
            return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)
    callbacks_list = [
        lr_callback,
        tensorboard_callback,
        early_stopping,
        model_checkpoint_callback,
        CSVLogger("training.csv")
    ]

    caption_model.fit(train_dataset,
                      epochs=epochs,
                      validation_data=valid_dataset,
                      callbacks=callbacks_list
                      )

    caption_model.save_weights("./weights/with_bleu/", save_format='tf')


# def inference(model):
#     # Select a random image from the validation dataset
#     # sample_img = np.random.choice(valid_images)
#     sample_img = valid_images[200]
#     sample_txt = valid_data[sample_img]
#
#     # plot
#     sample_img = read_image(sample_img)
#     img = sample_img.numpy().astype(np.uint8)
#     plt.imshow(img)
#     plt.show()
#
#     img = tf.expand_dims(sample_img, 0)
#
#     # inference
#     output = model(img)
#
#     # cleanup
#     sample_txt = [j.replace(" <end>", "") for j in sample_txt]
#     sample_txt = [item.replace("<start> ", "") for item in sample_txt]
#
#     # show results
#     print("PREDICTED CAPTION:", end=" ")
#     print(output.replace("<start> ", "").replace(" <end>", "").strip())
#     print("reference text :", sample_txt)
#     score = sentence_bleu(list(output.replace("<start> ", "").replace(" <end>", "").strip()), list(sample_txt))
#     print(score)


if __name__ == '__main__':
    main(epochs=90)

    # inference
    # model = ImageCaptioningModel(training=True)
    # model(tf.ones((1, 299, 299, 3)))
    # model.load_weights('image_caption_20.h5')
    # inference(model)
