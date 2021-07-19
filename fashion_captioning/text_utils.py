import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re


class Vectorization:
    def __init__(self, config):
        self.seq_length = config['seq_length']
        self.strip_chars = self._strip_chars()
        self.standardize = self._standardize
        self._vectorization = TextVectorization(output_mode='int',
                                                output_sequence_length=config['seq_length'],
                                                standardize=self.standardize,
                                                pad_to_max_tokens=True)

    def __call__(self, *args, **kwargs):
        return self._vectorization

    def _strip_chars(self):
        strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
        strip_chars = strip_chars.replace("<", "")
        strip_chars = strip_chars.replace(">", "")
        return strip_chars

    def _standardize(self, input_string):
        lowercase = tf.strings.lower(input_string)
        return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(self.strip_chars), "")

    def _adapt(self, text_data):
        self._vectorization.adapt(text_data)

    def get_vocabulary(self, text_data):
        self._adapt(text_data)
        vocab = self._vectorization.get_vocabulary()
        return vocab, len(vocab)

    def index_lookup(self, vocab):
        return dict(zip(range(len(vocab)), vocab))

    @property
    def vectorization(self):
        return self._vectorization

