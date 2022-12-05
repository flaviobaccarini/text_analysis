from text_analysis.read_write_data import read_data
from text_analysis.vectorize_data import get_vocabulary, tocat_encode_labels
from text_analysis.vectorize_data import init_vector_layer, vectorize_X_data_tf, calculate_max_len
from text_analysis.vectorize_data import flatten_unique_voc
import configparser
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.layers import Embedding
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def build_model(vocab_size: int,
                embedding_dim: int, 
                maxlen: int) -> tf.keras.models.Sequential:
    '''
    This function builds the neural network model.
    The three input parameters are used for the initialization
    of the embedding layer.

    Parameters:
    ===========
    vocab_size: int
                It is the vocabulary size. 
                Number of tokens present in the vocabulary.

    embedding_dim: int
                   Embedding vector size for the data after
                   the embedding layer.

    maxlen: int
            Input length of the data (before the embedding layer). 

    Returns:
    =========
    model: tf.keras.models.Sequential
           Neural network model.
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

def main():

    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('ANALYSIS', 'folder_name')
    dataset_folder = Path('preprocessed_datasets') / analysis_folder

    # READ DATA
    data = read_data(dataset_folder)
    df_train, df_valid, _ = data

    # READ FROM CONFIG FILE
    embedding_vector_size = int(config_parse.get('PARAMETERS_TRAIN', 'embedding_vector_size'))
    epochs = int(config_parse.get('PARAMETERS_TRAIN', 'epochs'))
    learning_rate = float(config_parse.get('PARAMETERS_TRAIN', 'learning_rate'))
    batch_size = int(config_parse.get('PARAMETERS_TRAIN', 'batch_size'))
    checkpoint_path = Path('checkpoint') / analysis_folder
    checkpoint_path.mkdir(parents = True, exist_ok = True)

    all_train_words = list(df_train['clean_text']) + list(df_valid['clean_text'])
    vocabulary = get_vocabulary(all_train_words)
    unique_vocabulary = flatten_unique_voc(vocabulary)
    maxlen = calculate_max_len(df_train['clean_text'])
    
    # VECTORIZE THE TEXT
    vectorize_layer = init_vector_layer(maxlen, unique_vocabulary)

    X_train = vectorize_X_data_tf(df_train['clean_text'], vectorize_layer)
    X_valid = vectorize_X_data_tf(df_valid['clean_text'], vectorize_layer)

    y_train = tocat_encode_labels(df_train['label'])
    y_valid = tocat_encode_labels(df_valid['label'])

    vocab_size = len(vectorize_layer.get_vocabulary())

    model = build_model(vocab_size = vocab_size,
                        embedding_dim = embedding_vector_size,
                        maxlen = maxlen)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # CHECKPOINT CALLBACK
    checkpoint_model_path = checkpoint_path / 'best_model.hdf5' 
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_model_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    # EARLY STOP CALLBACK
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

    # TRAIN
    history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_valid, y_valid),
    callbacks = [model_checkpoint_callback, early_stop_callback],
    batch_size=batch_size,
    epochs=epochs)

    hist_df = pd.DataFrame(history.history) 
    # save to csv: 
    hist_csv_file = checkpoint_path / 'history.csv'

    hist_df.to_csv(hist_csv_file, index = False)
    


if __name__ == '__main__':
    main()
