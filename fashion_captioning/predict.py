import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from train import vectorization, index_lookup, vocab_size, valid_data
from config import config
from models import TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel, get_cnn_model
from data_utils import read_image, load_captions_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

"""Models"""
cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=config['embed_dim'], dense_dim=config['ff_dim'],
                                  num_heads=config['num_attention_heads'])
decoder = TransformerDecoderBlock(embed_dim=config['embed_dim'], ff_dim=config['ff_dim'],
                                  num_heads=config['num_attention_heads'], seq_len=config['seq_length'],
                                  vocab_size=vocab_size)
caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
caption_model.load_weights(f"{config['weights_dir']}{config['dataset']}/{config['num_attention_heads']}_heads/")


def generate_caption(model, sample):
    img_path, sample_txt = sample
    sample_txt = [item.replace("<start> ", "") for item in sample_txt]
    sample_txt = [j.replace(" <end>", "") for j in sample_txt]

    # Read the image from the disk
    sample_img = read_image(img_path)

    # Pass the image to the CNN
    img = tf.expand_dims(sample_img, 0)
    img = model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(config['seq_length'] - 1):
        tokenized_caption = vectorization.vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = model.decoder(tokenized_caption, encoded_img,
                                    training=False, mask=mask)

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == " <end>":
            break
        decoded_caption += " " + sampled_token
    sample_txt = [item.replace("<start> ", "") for item in sample_txt]
    decoded_caption = decoded_caption.replace("<start> ", "").replace(" <end>", "").strip()
    return sample_txt, decoded_caption


def test_on_sample(model):
    # Check predictions for a few samples
    import random
    sample = random.choice(list(valid_data.items()))
    true_txt, pred_txt = generate_caption(model, sample)
    # sample_txt = sample_txt.replace("<start> ", "").replace(" <end>", "").strip()
    print('Individual 1-gram: %f' % sentence_bleu(pred_txt, true_txt, weights=(1, 0, 0, 0)))
    print('Individual 2-gram: %f' % sentence_bleu(pred_txt, true_txt, weights=(0, 1, 0, 0)))
    print('Individual 3-gram: %f' % sentence_bleu(pred_txt, true_txt, weights=(0, 0, 1, 0)))
    print('cumulative 4-gram: %f' % sentence_bleu(pred_txt, true_txt, weights=(0.25, 0.25, 0.25, 0.25)))

    print("PREDICTED CAPTION:", end=" ")
    print(pred_txt.replace("<start> ", "").replace(" <end>", "").strip())
    print("reference text :", true_txt)


def eval_on_test_set(model):
    captions_imgs, texts = load_captions_data(f"{config['annotations_dir']}test_{config['dataset']}.txt")
    sum_bleu_score = 0
    for i, sample in enumerate(captions_imgs.items()):
        sample_txt, pred_txt = generate_caption(model, sample)
        bleu_score = sentence_bleu([sample_txt[0].split()], pred_txt.split(), weights=(1, 0, 0, 0))
        if bleu_score > 0:
            sum_bleu_score += bleu_score
            print(f'{i}: {bleu_score}')
    avg_bleu = sum_bleu_score / len(captions_imgs.items())
    return avg_bleu


if __name__ == '__main__':
    # avg_bleu = eval_on_test_set(caption_model)
    # print(f'avg_blue is {avg_bleu}')
    true_txt, pred_txt = generate_caption(model=caption_model, sample=('images/81096.jpg', ['<start> animal print ruffle sleeve top <end>']))
    print('Individual 1-gram: %f' % sentence_bleu(true_txt, pred_txt, weights=(1, 0, 0, 0)))
    print('Individual 2-gram: %f' % sentence_bleu(true_txt, pred_txt, weights=(0, 1, 0, 0)))
    print('Individual 3-gram: %f' % sentence_bleu(true_txt, pred_txt, weights=(0, 0, 1, 0)))
    print('cumulative 4-gram: %f' % sentence_bleu(true_txt, pred_txt, weights=(0.25, 0.25, 0.25, 0.25)))

    print("PREDICTED CAPTION:", end=" ")
    print(pred_txt.replace("<start> ", "").replace(" <end>", "").strip())
    print("reference text :", true_txt)