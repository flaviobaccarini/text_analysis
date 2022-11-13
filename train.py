from read_write_data import read_data
from vectorize_data import get_vocabulary, tocat_encode_labels, vectorize_X_y_data
import configparser
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

#for model-building
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import pandas as pd

from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import pickle

from prediction_results import prediction, visualize_results

def build_model(vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim, 
                        input_length=maxlen))
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



def train_classificator(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf


def train_neural_network(data,
                        embedding_vector_size,
                        batch_size,
                        epochs,
                        learning_rate,
                        checkpoint_path,
                        analysis):

    df_train, df_valid, df_test = data

    # PREPARE THE TOKENIZER FOR THE DATA                    
    word_count = [len(str(words).split()) for words in df_train['clean_text']]
    maxlen = int(np.mean(word_count) + 3*np.std(word_count))

    max_num_words = int(len(get_vocabulary((df_train['clean_text'],
                                    df_valid['clean_text'],
                                    df_test['clean_text']))) * 1.5)
    tokenizer = tensorflow_tokenizer(max_num_words = max_num_words, 
                                    text = df_train['clean_text'])

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    # PREPARE THE DATA
    X_train = from_text_to_X_vector(df_train['clean_text'], tokenizer, maxlen)
    X_valid = from_text_to_X_vector(df_valid['clean_text'], tokenizer, maxlen)

    y_train = tocat_encode_labels(df_train['label'])
    y_valid = tocat_encode_labels(df_valid['label'])

    # BUILD THE MODEL
    model = build_model(vocab_size = vocab_size,
                        embedding_dim = embedding_vector_size,
                        maxlen = maxlen)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # CHECKPOINT CALLBACK
    checkpoint_folder = Path(checkpoint_path)
    checkpoint_folder = checkpoint_folder / analysis
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_filepath = checkpoint_folder / 'best_model.hdf5'
    model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    # EARLY STOP CALLBACK
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=20)

    # TRAIN
    history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_valid, y_valid),
    callbacks = [model_checkpoint_callback, early_stop_callback],
    batch_size=batch_size,
    epochs=epochs)
    
    # SAVE THE MODEL AND THE HISTORY in a DATAFRAME

    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    # save to csv: 
    hist_csv_file = checkpoint_folder / 'history.csv'

    hist_df.to_csv(hist_csv_file, index = False)
    

def train_logistic_regressor(data,
                            embedding_vector_size,
                            min_count,
                            random_state,
                            checkpoint_path,
                            analysis):
    df_train, df_valid, df_test = data
    vocabulary = get_vocabulary((df_train['clean_text'], df_valid['clean_text'], df_test['clean_text']))
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5,
                         min_count=min_count, workers=1, seed = random_state)   

    # PREPARE THE DATA
    df_train_val = pd.concat([df_train, df_valid], ignore_index = True)
    X_train, y_train = vectorize_X_y_data((df_train_val['clean_text'], df_train_val['label']), modelw2v)
    X_test, y_test = vectorize_X_y_data((df_test['clean_text'], df_test['label']), modelw2v)

    # TRAIN
    print("Train logistic regressor...")
    lr_w2v = LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2', max_iter=50)
    lr_w2v = train_classificator(lr_w2v, X_train, y_train)
    print("Logistic Regression: training done")

    # SAVING MODEL
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path = checkpoint_path / analysis
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    file_path = checkpoint_path / 'logistic_regression.sav'
    with open(file_path, 'wb') as file:
        pickle.dump(lr_w2v, file)

def main():
    
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('INPUT_OUTPUT', 'analysis')
    dataset_folder = 'preprocessed_datasets'

    data = read_data(dataset_folder, analysis_folder)
    embedding_vector_size = int(config_parse.get('PARAMETERS_TRAIN', 'embedding_vector_size'))
    epochs = int(config_parse.get('PARAMETERS_TRAIN', 'epochs'))
    learning_rate = float(config_parse.get('PARAMETERS_TRAIN', 'learning_rate'))
    batch_size = int(config_parse.get('PARAMETERS_TRAIN', 'batch_size'))
    min_count_words_w2v = int(config_parse.get('PARAMETERS_TRAIN', 'min_count_words_w2v'))
    random_state = int(config_parse.get('PREPROCESS', 'seed'))
    
    train_logistic_regressor(data, embedding_vector_size, min_count = min_count_words_w2v,
                            checkpoint_path='checkpoint', analysis = analysis_folder, random_state=random_state)
 

    train_neural_network(data, embedding_vector_size, batch_size = batch_size,
                        epochs = epochs, learning_rate = learning_rate,
                        checkpoint_path = 'checkpoint', analysis = analysis_folder)
    
if __name__ == '__main__':
    main()
