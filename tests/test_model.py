from binary_classifier.model import build_model


def test_build_model():
    '''
    Test function to test the working of build_model.
    The function build_model takes as input four variables:
    vocab_size, emdebbing_vector_size, maxlen and neurons.
    The first three variables are used for the creation of the 
    Embedding layer. The last input (neurons) has to be a list of
    at least two integer numbers, which represents the neuron numbers 
    of the first and the last layers.

    At least five layers are generated (for example neurons = [64, 1]):
    1) embedding layer 
    2) bidirection LSTM -> 64 weights
    3) batch normalization
    4) dropout
    5) dense -> 1 weight, sigmoid activation

    If neurons is [64, 32, 16, 1] the layers are:
    1) embedding layer 
    2) bidirection LSTM -> 64 weights
    3) batch normalization
    4) dropout
    5) dense -> 32 weights
    6) dropout
    7) dense -> 16 weights
    8) dropout
    9) dense -> 1 weight, sigmoid activation

    It is important that the last number in the neurons list is 1,
    because it's a binary classifier with sigmoid activation.
    '''
    vocab_size = 10000
    embedding_vector_size = 32
    maxlen = 10
    neurons = [64, 1]
    model = build_model(vocab_size, embedding_vector_size, maxlen,
                        neurons)
    # embedding, bidirect lstm, batch normalization, dropout, dense
    assert( len(model.layers) == 5)

    neurons = [64, 32, 16, 1]
    model = build_model(vocab_size, embedding_vector_size, maxlen,
                        neurons)
    # embedding, bidirect lstm, batch normalization, dropout
    # (dense, dropout) * 2, dense
    assert( len(model.layers) == 9)


