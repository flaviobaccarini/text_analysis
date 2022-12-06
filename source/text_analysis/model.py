'''
MODEL MODULE
=============
In this module there is the function for building the neural network module.
'''
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.layers import Embedding

def build_model(vocab_size: int,
                embedding_dim: int, 
                maxlen: int) -> tf.keras.models.Sequential:
    '''
    This function builds the neural network model. The model is
    a Bidirectional LSTM with a 8 layers:
    1) Embedding layer
    2) Bidrectional LSTM -> 64 weights
    4) Dropout
    5) Dense layer -> 32 weigths
    6) Dropout
    7) Dense layer -> 10 weights
    8) Dropout
    9) Dense layer -> 1 weight
    The three input parameters are used for the initialization
    of the embedding layer.

    Parameters:
    ===========
    vocab_size: int
                It is the vocabulary size. 
                Number of tokens present in the vocabulary.

    embedding_dim: int
                   Embedding vector size of the data after
                   the embedding layer.

    maxlen: int
            Input length of the data (before the embedding layer). 

    Returns:
    =========
    model: tf.keras.models.Sequential
           Bidirectional LSTM sequential model.
    '''
    
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim, 
                        input_length=maxlen))
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0)))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dropout(0.25))    
    model.add(Dense(1, activation='sigmoid'))

    return model
