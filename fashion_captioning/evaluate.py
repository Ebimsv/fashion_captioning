import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras as keras

from .text_utils import Vectorization
from .data_utils import load_captions_data, make_dataset
from .models import get_cnn_model, TransformerEncoderBlock, TransformerDecoderBlock, ImageCaptioningModel
from .config import config

""" Preparing the dataset"""
captions_mapping, text_data = load_captions_data(f"{config['annotations_dir']}test_{config['dataset']}.txt")
_, all_text_data = load_captions_data(f"{config['annotations_dir']}train_{config['dataset']}.txt")

print("Number of training samples: ", len(captions_mapping.items()))

""" Vectorization """
vectorization = Vectorization(config)
vocab, vocab_size = vectorization.get_vocabulary(all_text_data)
index_lookup = vectorization.index_lookup(vocab)

test_dataset = make_dataset(list(captions_mapping.keys()), list(captions_mapping.values()), vectorization=vectorization)

""" Build Models """
cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=config['embed_dim'], dense_dim=config['ff_dim'],
                                  num_heads=config['num_attention_heads'])
decoder = TransformerDecoderBlock(embed_dim=config['embed_dim'], ff_dim=config['ff_dim'],
                                  num_heads=config['num_attention_heads'], seq_len=config['seq_length'],
                                  vocab_size=vocab_size)
caption_model = ImageCaptioningModel(cnn_model=cnn_model, encoder=encoder, decoder=decoder)
cross_entropy = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
caption_model.compile(optimizer=keras.optimizers.Adam(), loss=cross_entropy)

if __name__ == '__main__':
    weights_path = f"{config['weights_dir']}{config['dataset']}/{config['num_attention_heads']}_heads/"
    caption_model.load_weights(weights_path)

    """ Evaluation """
    caption_model.evaluate(test_dataset)
