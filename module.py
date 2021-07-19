import numpy as np
import base64
import io
from PIL import Image
import tensorflow as tf
from fashion_captioning import text_utils
from fashion_captioning.config import config
from fashion_captioning.train import index_lookup, vocab_size, vectorization
from fashion_captioning.models import TransformerEncoderBlock, \
    TransformerDecoderBlock, ImageCaptioningModel, get_cnn_model


def bytes_img(image: Image):
    data = io.BytesIO()
    image = image.convert('RGB')
    image.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())
    return encoded_img_data.decode('utf-8')


"""Models"""
cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=config['embed_dim'], dense_dim=config['ff_dim'],
                                  num_heads=config['num_attention_heads'])
decoder = TransformerDecoderBlock(embed_dim=config['embed_dim'], ff_dim=config['ff_dim'],
                                  num_heads=config['num_attention_heads'], seq_len=config['seq_length'],
                                  vocab_size=vocab_size)
caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
caption_model.load_weights(f"{config['weights_dir']}{config['dataset']}/{config['num_attention_heads']}_heads/")


def caption_image(image):
    # img = np.array(image)
    # img = tf.convert_to_tensor(img)
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = tf.image.resize(img, config["image_size"])
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.expand_dims(img, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(config['seq_length'] - 1):
        tokenized_caption = vectorization.vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(tokenized_caption, encoded_img,
                                            training=False, mask=mask)

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token
    decoded_caption = decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()
    return {'Text': decoded_caption}
