from nltk import word_tokenize
import nltk
import numpy as np
from sklearn import preprocessing


def vectorize_X_y_data(data, model):

    clean_text, labels = data
    X_tok = [word_tokenize(words) for words in clean_text]  
 
    w2v = dict(zip(model.wv.index_to_key, model.wv.vectors)) 

    modelw = MeanEmbeddingVectorizer(w2v)
    # converting text to numerical data using Word2Vec
    X_vectors_w2v = modelw.transform(X_tok)

    y_vector_categorical = tocat_encode_labels(labels)

    return X_vectors_w2v, y_vector_categorical


def tocat_encode_labels(labels):
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    y_vector_categorical = le.transform(labels)
    return y_vector_categorical

def get_vocabulary(list_words):
    vocabulary = []
    for words in list_words:
        vocabulary.append(words)
    vocabulary = [words for sublist_word in vocabulary for words in sublist_word]
    
    #vocabulary = [vocabulary.append(words) for words in list_words]
    vocabulary = [nltk.word_tokenize(words) for words in vocabulary]

    return vocabulary



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
