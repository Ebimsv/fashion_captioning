import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu


def calculate_bleu(y_true, y_pred):
    bleu = tf.py_function(func=preprocess,
                          inp=[y_true, y_pred],
                          Tout=[tf.float32])
    return bleu


def preprocess(y_true, y_pred):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    y_pred = np.argmax(y_pred, axis=2)

    bleu_scores = []
    for index, _ in enumerate(y_pred):
        bleu_scores.append([sentence_bleu([y_true[index]], y_pred[index], weights=(0.5, 0.5, 0, 0))])

    return np.mean(bleu_scores).astype(np.float32)
