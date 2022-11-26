from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding, BatchNormalization

def build_model(vocab_size, embedding_dim, maxlen, neurons):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim, 
                        input_length=maxlen))
    model.add(Bidirectional(LSTM(neurons[0], recurrent_dropout=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    for nr_neuron in neurons[1:-1]:
        model.add(Dense(nr_neuron))
        model.add(Dropout(0.25))
    model.add(Dense(neurons[-1], activation='sigmoid'))


    return model
