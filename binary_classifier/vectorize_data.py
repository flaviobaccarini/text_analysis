'''
VECTORIZE MODULE
=================
Functions for vectorization of the text data.
'''
from nltk import word_tokenize
import nltk
import numpy as np
from numpy.typing import ArrayLike
from sklearn import preprocessing
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf
from gensim.models import Word2Vec

def vectorize_X_data_lr(text_data: ArrayLike,
                        model: Word2Vec) -> np.ndarray:
    '''
    Function for vectorizing data for the logistic regressor.
    It takes as input text_data, which is 1-D array-like of 
    strings. This array contains all the sentences, that will
    be vectorized by this function thank to a Word2Vec model.
    In the end the vectorized text is a 2-D numpy array with
    shape (len(text_data), dim), where dim is equal to the 
    vector_size value of the Word2Vec model.

    Parameters:
    ===========
    text_data: 1-D array-like[str]
               Sequence that contains all the text and sentences.
    
    model: gensim.models.Word2Vec
           Word2Vec model already trained.
           This is useful, because it maps each single word, present
           in the vocabulary of the model, to a vector of floats.

    Returns:
    =========
    X_vectors_w2v: 2-D np.array[floats]
                   Vectorized text data.
                   The array has a shape of (len(text_data), dim),
                   where dim is vector_size of the Word2Vec model.
                   The values of the array are positive or negative
                   floats.
    '''
    X_tok = [word_tokenize(words) for words in text_data]  
    w2v = dict(zip(model.wv.index_to_key, model.wv.vectors)) 
    modelw = MeanEmbeddingVectorizer(w2v)
    # converting text to numerical data using Word2Vec
    X_vectors_w2v = modelw.transform(X_tok)
    return X_vectors_w2v

def tocat_encode_labels(labels: ArrayLike,
                        classes:bool = False) -> np.ndarray:
    '''
    Function to trasnform the original labels (which
    could be string, integers...) into
    a categorical encoded labels (integer labels).

    Parameters:
    ============
    labels: 1-D array-like
            Sequence of the labels that have to be 
            categorical encoded.
    
    classes: bool default: False
             If this bool is true the function returns
             both the categorical encoded labels and the
             classes. If it is false the function returns
             only the categorical encoded labels.
    
    Returns:
    =========
    y_vector_categorical: 1-D np.array
                          Labels categorical encoded
    
    uniq_labels: 1-D np.array
                 It is returned only if the classes bool 
                 variable is True.
                 It is a sequence contained the original 
                 unique labels.
    '''
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    y_vector_categorical = le.transform(labels)
    
    if classes is True:
        uniq_labels = le.classes_
        return y_vector_categorical, uniq_labels
    
    return y_vector_categorical


def get_vocabulary(sentences: ArrayLike) -> list[list[str]]:
    '''
    Function for getting the vocabulary (all single words),
    starting from all the sentences.
    Example: 
    sentences = ['hi, how are you?', 'fine thanks']
    vocabulary = [ ['hi', 'how', 'are', 'you', '?'],
                  ['fine', 'thanks'] ]

    Parameters:
    ===========s
    sentences:  1-D array-like[str]
                Sequence of sentences. From these sentences
                the function will extrapolate the vocabulary.
                
    Returns:
    =========
    vocabulary: list[list[str]]
                The list contains a number of lists equal
                to the number of sentences (len(sentences)).
                Each sublist contains all the single words 
                of the sentence tokenized.
    '''
    vocabulary = [nltk.word_tokenize(words) for words in sentences]
    return vocabulary

def flatten_unique_voc(vocabulary_lists: list[list[str]]) -> ArrayLike:
    '''
    Function for flattening and getting the unique words
    from a list of words which represents the initial
    vocabulary (the output of the get_vocabulary function).

    Parameters:
    ===========
    vocabulary_lists: list[list[str]]
                      The list contains a number of lists.
                      Each single sublist corresponds to an
                      one tokenized sentence.
                     
    Returns:
    =========
    voc_unique: 1-D np.array[str]
                The array contains all the unique tokenized
                words from the vocabulary_lists.

    Example:
    =========
    vocabulary_lists = [ ['hi', 'how', 'are', 'you', '?'],
                  ['fine', 'thanks', 'and', 'how', 'are', 'you', '?'] ]
    voc_unique = ['hi', 'how', 'are', 'you', '?', 'fine, 'thanks', 'and]
    '''
    vocabulary_flatten = [word for sentence in vocabulary_lists 
                            for word in sentence]
    voc_unique = np.unique(vocabulary_flatten)
    return voc_unique
    

#building Word2Vec model
class MeanEmbeddingVectorizer():
    '''
    Class for the vectorization of the text data
    for the lostic regressor train/test.
    '''
    def __init__(self, word2vec: dict):
        '''
        Initialize function for the class.
        Two main parameters are initialized: 
        1) the vocabulary, where each single word
           is mapped to the corresponding vector.
        2) the dimension of the final vector.

        Parameters:
        ============
        word2vec: dict
                  Dictionary that maps each word to each
                  float vector. This dictionary could be 
                  the result of a Word2Vec trained model.  

        Initialize:
        ============
        word2vec: dict
                  This dictionary is initialized with the
                  one given as input. His job is to map
                  all the single word to their corresponding 
                  vector.
        dim: int
             Length of the final output vector. 
        '''
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
            
    def transform(self, X_text: ArrayLike) -> np.ndarray:
        '''
        Function for the vectorization of the text data for the
        logistic regressor model.
        The function takes as input X_text, which is a 1-D array-like
        of strings, that contains all the sentences. For each sentence
        this function computes the average for the words of sentence 
        present in the dictionary (word2vec). 
        If a sentence contains words with no mapping in the dictionary,
        the function returns a vector of zeros with the same dimensionality
        as all the other vectors.
        In the end the shape of the vector is: (len(X_text), self.dim), where
        self.dim is the dimensionality of all the vectors (to understand better
        what is the value of self.dim, please see the function vectorize_X_data_lr).

        Parameters:
        ============
        X_text: 1-D array-like[str]
                Sequence of the sentences to vectorize and convert 
                in numbers.

        Returns:
        ========
        X_vector: 2-D np.array[floats]
                  The vectorized text.
                  The text is converted to float numbers thanksì
                  to the self.word2vec dictionary, that maps each 
                  single word present in the dictionary to a vector
                  of floats.
                  The shape is (len(X_text, self.dim)).
        '''
        X_vector = np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in X_text
        ])
        return X_vector


def calculate_max_len(text: ArrayLike) -> int:
    '''
    Function for calculating the length to pad/truncate
    the sequences. The padding/truncation operation is 
    performed in order to have vector data with the same 
    shape (necessary for the neural network).
    In this function the average number of words for each single
    sentence is computed. The standard deviation of the 
    number of words is computed too. The ouput length is 
    equal to the average number + 2*standard deviation 
    computed before.

    Parameters:
    ============
    text: 1-D array-like[str]
          The sequence with all the sentences.
    
    Returns:
    =========
    maxlen: int
            It represents the maximum length for the vector
            sequence. It is computed from the average number
            and the standard deviation of the number of words
            in the sentences (average + 2*std).
    '''
    helper_vocabulary = get_vocabulary(text)
    word_count = [len(sentence) for sentence in helper_vocabulary]
    maxlen = int(np.round(np.mean(word_count) + 2*np.std(word_count)))
    return maxlen

def init_vector_layer(maxlen: int,
                      uniq_vocabulary: ArrayLike) -> tf.keras.layers.TextVectorization:
    '''
    Function for initializing the vectorize layer.
    This vectorize layer is then used for the vectorization
    of the text data in order to be ready as input for a 
    neural network.

    Parameters:
    ===========
    maxlen: int
            It represents the maximum length for the vector
            sequence (padding/truncation operation).

    uniq_vocabulary: 1-D array-like[str]
                     It represents the unique vocabulary.
                     The list contains all the unique tokens
                     (unique single words). The vocabulary 
                     should be composed only by training words
                     (no test words).

    Returns:
    =========
    vectorize_layer: tf.keras.layers.TextVectorization
                     The vectorize layer, ready to vectorize
                     the text data.
    '''
    vectorize_layer = TextVectorization(
    standardize=None,
    output_mode='int',
    output_sequence_length=maxlen,
    vocabulary = uniq_vocabulary
    )
    return vectorize_layer

def vectorize_X_data_tf(text: ArrayLike,
                        vector_layer: tf.keras.layers.TextVectorization) -> tf.Tensor:
    '''
    Function for vectorization of text data for the neural network.
    It converts the text, which could be a single string (a single sentence)
    or a 1-D array-like string (all the sentences) into vectors of integers
    according to the vocabulary from the vector_layer.
    vector_layer is a TextVectorization layer ready for the vectorization
    of the text.

    Parameters:
    ============
    text: 1-D array-like[str] or str
          If it is 1-D array-like[str] it is a sequence of all the sentences.
          If instead it is a simple string, it is a single sentence.

    vector_layer: tf.keras.layers.TextVectorization
                  TextVectorization layer ready for the vectorization.

    Returns:
    ========
    vectorized_text: tf.Tensor
                     The vectorized text.
                     The shape is (len(text), maxlen) if text
                     is a 1-D array-like (to understand what maxlen
                     is see functions calculate_max_len and
                     init_vector_layer).
                     The shape is (1, maxlen) if text is a single string.
                     The tensor is composed by 32 bit integers.
    '''
    text = tf.convert_to_tensor(text)
    vectorized_text = tf.cast(vector_layer(text), tf.int32)
    return vectorized_text
