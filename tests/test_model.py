from binary_classifier.model import build_model
import tensorflow as tf

def test_build_model():
    '''
    Test function to test the working of build_model.
    The function build_model takes as input four variables:
    vocab_size, emdebbing_vector_size, maxlen.
    The first three variables are used for the creation of the 
    embedding layer. 
    8 layers are generated :
    1) embedding layer 
    2) bidirection LSTM -> 64 weights
    3) dropout
    4) dense -> 32 weights
    5) dropout
    6) dense -> 10 weights
    7) dropout
    8) dense -> 1 weight
    '''
    vocab_size = 10000
    embedding_vector_size = 32
    maxlen = 10
    model = build_model(vocab_size, embedding_vector_size, maxlen)
    assert( len(model.layers) == 8)
    assert(type(model.layers[0]) == tf.keras.layers.Embedding)
    assert(type(model.layers[1]) == tf.keras.layers.Bidirectional)
    assert(type(model.layers[2]) == tf.keras.layers.Dropout)
    assert(type(model.layers[3]) == tf.keras.layers.Dense)
    assert(type(model.layers[4]) == tf.keras.layers.Dropout)
    assert(type(model.layers[5]) == tf.keras.layers.Dense)
    assert(type(model.layers[6]) == tf.keras.layers.Dropout)
    assert(type(model.layers[7]) == tf.keras.layers.Dense)
