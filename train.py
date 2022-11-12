from read_write_data import read_data
from vectorize_data import get_vocabulary, tocat_encode_labels
import configparser
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding, GlobalMaxPool1D
#from tensorflow.keras.layers.embeddings import Embeddingà
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

#for model-building
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd

def build_model(vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                            output_dim=embedding_dim, 
                            input_length=maxlen))
    #model.add(GlobalMaxPool1D())
    #model.add(Dense(32, activation='relu'))
    #model.add(Dropout(0.3))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(32, recurrent_dropout=0)))
    model.add(Dropout(0.3))
    model.add(Dense(16))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model


def tensorflow_tokenizer(max_num_words, text):
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(text)
    return tokenizer

def from_text_to_X_vector(text, tokenizer, maxlen):
    X = np.array(text)
    X = tokenizer.texts_to_sequences(X)
    X = pad_sequences(X, padding='post', maxlen=maxlen)
    return X

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    input_folder = config_parse.get('INPUT_OUTPUT', 'folder_preprocessed_datasets')

    df_train, df_valid, df_test = read_data(input_folder=input_folder)
    embedding_vector_size = int(config_parse.get('PARAMETERS_TRAIN', 'embedding_vector_size'))

    word_count = [len(str(words).split()) for words in df_train['clean_text']]
    maxlen = int(np.mean(word_count) + 3*np.std(word_count))

    max_num_words = int(len(get_vocabulary((df_train['clean_text'],
                                    df_valid['clean_text'],
                                    df_test['clean_text']))) * 1.5)
    tokenizer = tensorflow_tokenizer(max_num_words = max_num_words, text = df_train['clean_text'])

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    model = build_model(vocab_size = vocab_size,
                        embedding_dim = embedding_vector_size,
                        maxlen = maxlen)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    X_train = from_text_to_X_vector(df_train['clean_text'], tokenizer, maxlen)
    X_valid = from_text_to_X_vector(df_valid['clean_text'], tokenizer, maxlen)
    X_test = from_text_to_X_vector(df_test['clean_text'], tokenizer, maxlen)


    y_train = tocat_encode_labels(df_train['label'])
    y_valid = tocat_encode_labels(df_valid['label'])
    y_test = tocat_encode_labels(df_test['label'])

    checkpoint_folder = Path('checkpoint')
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_filepath = checkpoint_folder / 'best_model.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_valid, y_valid),
    callbacks = [model_checkpoint_callback, early_stop_callback],
    batch_size=64,
    epochs=200)
    
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    plot_history(history)
    
    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    # save to csv: 
    hist_csv_file = checkpoint_folder / 'history.csv'
    hist_df.to_csv(hist_csv_file, index = False)
if __name__ == '__main__':
    main()
