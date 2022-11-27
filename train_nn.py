from binary_classifier.read_write_data import read_data
from binary_classifier.vectorize_data import get_vocabulary, tocat_encode_labels
from binary_classifier.vectorize_data import init_vector_layer, vectorize_text_tf, calculate_max_len
from binary_classifier.vectorize_data import flatten_unique_voc
from binary_classifier.model import build_model
import configparser
import sys
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def main():

    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('INPUT_OUTPUT', 'analysis')
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

    all_train_words = list(df_train['clean_text']) + list(df_valid['clean_text'])
    vocabulary = get_vocabulary(all_train_words)

    # TOKENIZE THE TEXT
    maxlen = calculate_max_len(df_train['clean_text'])
    unique_vocabulary = flatten_unique_voc(vocabulary)

    vectorize_layer = init_vector_layer(maxlen, unique_vocabulary)

    X_train = vectorize_text_tf(df_train['clean_text'], vectorize_layer)
    X_valid = vectorize_text_tf(df_valid['clean_text'], vectorize_layer)

    y_train, _ = tocat_encode_labels(df_train['label'])
    y_valid, _ = tocat_encode_labels(df_valid['label'])

    vocab_size = len(vectorize_layer.get_vocabulary()) + 1

    model = build_model(vocab_size = vocab_size,
                        embedding_dim = embedding_vector_size,
                        maxlen = maxlen,
                        neurons = [64, 32, 16, 1])
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
