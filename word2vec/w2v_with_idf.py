import pickle
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import pairwise_distances
from word2vec.utility import load_image, data, idf_title_vectorizer, idf_title_features
import seaborn as sns

'''
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
'''

with open('word2vec_model', 'rb') as handle:
    model = pickle.load(handle)


# Utility functions

def get_word_vec(sentence, doc_id, m_name):
    vec = []
    for i in sentence.split():
        if i in vocab:
            if m_name == 'weighted' and i in idf_title_vectorizer.vocabulary_:
                vec.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[i]] * model[i])
            elif m_name == 'avg':
                vec.append(model[i])
        else:
            vec.append(np.zeros(shape=(300,)))

    return np.array(vec)


def get_distance(vec1, vec2):
    final_dist = []
    for i in vec1:
        dist = []
        for j in vec2:
            dist.append(np.linalg.norm(i - j))
        final_dist.append(np.array(dist))
    return np.array(final_dist)


def fig2img(fig) -> Image:
    """Convert a Matplotlib figure to a PIL Image and return it"""
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def heat_map_w2v(sentence1, sentence2, url, doc_id1, doc_id2, model):
    s1_vec = get_word_vec(sentence1, doc_id1, model)
    s2_vec = get_word_vec(sentence2, doc_id2, model)
    s1_s2_dist = get_distance(s1_vec, s2_vec)
    gs = gridspec.GridSpec(1, 1, width_ratios=[4], height_ratios=[2])
    fig = plt.figure(figsize=(8, 4))

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 8))
    ax = plt.subplot(gs[0])
    ax = sns.heatmap(np.round(s1_s2_dist, 4), annot=True)
    ax.set_xticklabels(sentence2.split())
    ax.set_yticklabels(sentence1.split())
    ax.set_title(sentence2)
    ax.grid(False)
    # display_img(url, ax, fig)
    pil_img = fig2img(plt.gcf())
    return pil_img


vocab = model.keys()


def build_avg_vec(sentence, num_features, doc_id, m_name):
    featureVec = np.zeros((num_features,), dtype="float32")

    nwords = 0

    for word in sentence.split():
        nwords += 1
        if word in vocab:
            if m_name == 'weighted' and word in idf_title_vectorizer.vocabulary_:
                featureVec = np.add(featureVec,
                                    idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[word]] * model[word])
            elif m_name == 'avg':
                featureVec = np.add(featureVec, model[word])
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


# IDF weighted Word2Vec for product similarity
doc_id = 0
feature_vectors = []
for i, title in enumerate(data['text']):
    feature_vectors.append(build_avg_vec(title, 300, i, 'weighted'))
feature_vectors = np.array(feature_vectors)


def weighted_w2v_model(sentence, num_results=3):
    # doc_id = list(data['text']).index(sentence)
    # id2sentence = data['text'].loc[doc_id]
    sentence_feature_vec = build_avg_vec(sentence, 300, None, 'avg')
    pairwise_dist = pairwise_distances(feature_vectors, sentence_feature_vec.reshape(1, -1))
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists = np.sort(pairwise_dist.flatten())[0:num_results]
    df_indices = list(data.index[indices])

    most_similars = []
    for i in range(0, len(indices)):
        pred_sentence = data['text'].loc[df_indices[0]]
        candidate_sentence = data['text'].loc[df_indices[i]]
        image_url = data['path'].loc[df_indices[i]]
        distance = pdists[i]
        id_ = indices[0]
        candidate_doc_id = indices[i]
        heat_map_img = heat_map_w2v(pred_sentence, candidate_sentence,
                                    image_url, id_, candidate_doc_id, 'weighted')
        pil_img = load_image(image_url)
        most_similars.append((heat_map_img, pil_img, candidate_sentence, distance))
    return most_similars


# results = weighted_w2v_model('topstitch milano knit jacket ', 3)

