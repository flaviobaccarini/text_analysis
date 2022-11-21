from binary_classifier.read_write_data import read_data
from binary_classifier.vectorize_data import get_vocabulary, tocat_encode_labels, vectorize_X_data_lr
from binary_classifier.vectorize_data import init_vector_layer, vectorize_X_data_tf, calculate_max_len

import configparser
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

from pathlib import Path
import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import pickle

from tqdm import tqdm

def build_model(vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim, 
                        input_length=maxlen))
    model.add(Bidirectional(LSTM(64, recurrent_dropout=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(32))
    model.add(Dropout(0.25))
    model.add(Dense(10))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    return model


def train_classificator(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf


def train_neural_network(data,
                        embedding_vector_size,
                        batch_size,
                        epochs,
                        learning_rate,
                        checkpoint_path):

    df_train, df_valid, df_test = data

    # TOKENIZE THE TEXT
    maxlen = calculate_max_len(df_train['clean_text'])
    vocabulary = get_vocabulary((df_train['clean_text'], df_valid['clean_text']))
    vocabulary = np.unique(vocabulary)

    vectorize_layer = init_vector_layer(maxlen, vocabulary)

    X_train = [vectorize_X_data_tf(text, vectorize_layer) for text in tqdm(df_train['clean_text'])]
    X_valid = [vectorize_X_data_tf(text, vectorize_layer) for text in tqdm(df_valid['clean_text'])]
    X_train = tf.stack(X_train, axis=0) 
    X_valid = tf.stack(X_valid, axis=0)

    vocab_size = len(vectorize_layer.get_vocabulary()) + 1

    y_train, _ = tocat_encode_labels(df_train['label'])
    y_valid, _ = tocat_encode_labels(df_valid['label'])

    model = build_model(vocab_size = vocab_size,
                        embedding_dim = embedding_vector_size,
                        maxlen = maxlen)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # CHECKPOINT CALLBACK
    checkpoint_model_path = checkpoint_path / 'best_model.hdf5' 
    model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_model_path,
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
    hist_csv_file = checkpoint_path / 'history.csv'

    hist_df.to_csv(hist_csv_file, index = False)
    

def train_logistic_regressor(data,
                            modelw2v,
                            file_path
                            ):
    df_train, df_valid, df_test = data

    # PREPARE THE DATA
    df_train_val = pd.concat([df_train, df_valid], ignore_index = True)
    X_train = vectorize_X_data_lr(df_train_val['clean_text'], modelw2v)
    y_train, _ = tocat_encode_labels(df_train_val['label'])

    # TRAIN
    print("Train logistic regressor...")
    lr_w2v = LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2', max_iter=20)
    lr_w2v = train_classificator(lr_w2v, X_train, y_train)
    print("Logistic Regression: training done")

    # SAVING MODEL
    with open(file_path, 'wb') as file:
        pickle.dump(lr_w2v, file)

def main():
    
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('INPUT_OUTPUT', 'analysis')
    dataset_folder = Path('preprocessed_datasets') / analysis_folder

    data = read_data(dataset_folder)
    embedding_vector_size = int(config_parse.get('PARAMETERS_TRAIN', 'embedding_vector_size'))
    epochs = int(config_parse.get('PARAMETERS_TRAIN', 'epochs'))
    learning_rate = float(config_parse.get('PARAMETERS_TRAIN', 'learning_rate'))
    batch_size = int(config_parse.get('PARAMETERS_TRAIN', 'batch_size'))
    min_count_words_w2v = int(config_parse.get('PARAMETERS_TRAIN', 'min_count_words_w2v'))
    random_state = int(config_parse.get('PREPROCESS', 'seed'))
    checkpoint_path = Path('checkpoint') / analysis_folder

    vocabulary = get_vocabulary((data[0]['clean_text'], data[1]['clean_text']))
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5,
                         min_count=min_count_words_w2v, workers=1, seed = random_state)   

    checkpoint_path.mkdir(parents = True, exist_ok = True)
    file_path_model_w2v = checkpoint_path / 'word2vec.model'
    modelw2v.save(str(file_path_model_w2v))

    file_path_model_lr = checkpoint_path / 'logistic_regression.sav'

    train_logistic_regressor(data, modelw2v, file_path_model_lr)
    
    
    # NEURAL NETWORK TRAIN:         
    train_neural_network(data, embedding_vector_size, batch_size = batch_size,
                        epochs = epochs, learning_rate = learning_rate,
                        checkpoint_path = checkpoint_path)
                        

if __name__ == '__main__':
    main()
