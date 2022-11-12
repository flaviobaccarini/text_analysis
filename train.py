from read_write_data import read_data
from vectorize_data import get_vocabulary, tocat_encode_labels
import configparser
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding, GlobalMaxPool1D
#from tensorflow.keras.layers.embeddings import Embeddingà
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

#for model-building
import numpy as np

def build_model(vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                            output_dim=embedding_dim, 
                            input_length=maxlen))
    model.add(GlobalMaxPool1D())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model


def tensorflow_tokenizer(max_num_words, text):
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(text)
    return tokenizer

def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    input_folder = config_parse.get('INPUT_OUTPUT', 'folder_preprocessed_datasets')

    df_train, df_valid, df_test = read_data(input_folder=input_folder)
    embedding_vector_size = int(config_parse.get('PARAMETERS_TRAIN', 'embedding_vector_size'))

    word_count = [len(str(words).split()) for words in df_train['clean_text']]
    maxlen = int(np.mean(word_count) + 3*np.std(word_count))

    X_train = np.array(df_train['clean_text'])
    X_valid = np.array(df_valid['clean_text'])

    y_train = np.array(df_train['label'])
    y_valid = np.array(df_valid['label'])

    max_num_words = int(len(get_vocabulary((df_train['clean_text'],
                                    df_valid['clean_text'],
                                    df_test['clean_text']))) * 1.5)
    tokenizer = tensorflow_tokenizer(max_num_words = max_num_words, text = X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_valid = tokenizer.texts_to_sequences(X_valid)

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_valid = pad_sequences(X_valid, padding='post', maxlen=maxlen)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    model = build_model(vocab_size = vocab_size,
                        embedding_dim = embedding_vector_size,
                        maxlen = maxlen)
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    y_train = tocat_encode_labels(y_train)
    y_valid = tocat_encode_labels(y_valid)


    history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_valid, y_valid),
    batch_size=10,
    epochs=3)
    
    
if __name__ == '__main__':
    main()
