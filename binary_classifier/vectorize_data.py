from nltk import word_tokenize
import nltk
import numpy as np
from sklearn import preprocessing
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf

def vectorize_X_data_lr(data, model):

    clean_text = data
    X_tok = [word_tokenize(words) for words in clean_text]  

    w2v = dict(zip(model.wv.index_to_key, model.wv.vectors)) 
    modelw = MeanEmbeddingVectorizer(w2v)
    # converting text to numerical data using Word2Vec
    X_vectors_w2v = modelw.transform(X_tok)

    return X_vectors_w2v


def tocat_encode_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    y_vector_categorical = le.transform(labels)
    classes = le.classes_
    return y_vector_categorical, classes


def get_vocabulary(list_words):

    vocabulary = [nltk.word_tokenize(words) for words in list_words]

    return vocabulary

def flatten_unique_voc(vocabulary_lists):

    vocabulary_flatten = [word for sentence in vocabulary_lists 
                            for word in sentence]
    voc_unique = np.unique(vocabulary_flatten)

    return voc_unique
    

#building Word2Vec model
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
    def fit(self, X, y):
        return self
            
    def transform(self, X):
        new_X = np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in X
        ])
            

        return new_X


def calculate_max_len(text):
    word_count = [len(str(words).split()) for words in text]
    maxlen = np.round(np.mean(word_count) + 2*np.std(word_count))
    return int(maxlen)

def init_vector_layer(maxlen, vocabulary):
    vectorize_layer = TextVectorization(
    standardize=None,
    output_mode='int',
    output_sequence_length=maxlen,
    vocabulary = vocabulary
    )
    return vectorize_layer

def vectorize_X_data_tf(text, vector_layer):
  text = tf.convert_to_tensor(text)
  return tf.cast(vector_layer(text), tf.int32)
