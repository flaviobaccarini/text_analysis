from binary_classifier.read_write_data import read_data
from binary_classifier.vectorize_data import get_vocabulary, tocat_encode_labels, vectorize_X_data_lr
from binary_classifier.vectorize_data import init_vector_layer, vectorize_X_data_tf, calculate_max_len
from binary_classifier.vectorize_data import flatten_unique_voc
from binary_classifier.train_models import train_neural_network, write_history, build_model
from binary_classifier.train_models import train_logistic_regressor
import configparser
import sys

import tensorflow as tf

from pathlib import Path
import pandas as pd

from gensim.models import Word2Vec

from tqdm import tqdm


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
    min_count_words_w2v = int(config_parse.get('PARAMETERS_TRAIN', 'min_count_words_w2v'))
    random_state = int(config_parse.get('PREPROCESS', 'seed'))
    checkpoint_path = Path('checkpoint') / analysis_folder

    # LOGISTIC REGRESSION

    all_train_words = list(df_train['clean_text']) + list(df_valid['clean_text'])
    vocabulary = get_vocabulary(all_train_words)

    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5,
                         min_count=min_count_words_w2v, workers=1, seed = random_state)   

    # PREPARE THE DATA:
    df_train_val = pd.concat([df_train, df_valid], ignore_index = True)
    X_train = vectorize_X_data_lr(df_train_val['clean_text'], modelw2v)
    y_train, _ = tocat_encode_labels(df_train_val['label'])

    # CHECKPOINT PATH:
    checkpoint_path.mkdir(parents = True, exist_ok = True)
    file_path_model_w2v = checkpoint_path / 'word2vec.model'
    modelw2v.save(str(file_path_model_w2v))

    file_path_model_lr = checkpoint_path / 'logistic_regression.sav'

    train_logistic_regressor(X_train, y_train, file_path_model_lr)



    # NEURAL NETWORK:

    # TOKENIZE THE TEXT
    maxlen = calculate_max_len(df_train['clean_text'])
    unique_vocabulary = flatten_unique_voc(vocabulary)

    vectorize_layer = init_vector_layer(maxlen, unique_vocabulary)

    X_train = [vectorize_X_data_tf(text, vectorize_layer) for text in tqdm(df_train['clean_text'])]
    X_valid = [vectorize_X_data_tf(text, vectorize_layer) for text in tqdm(df_valid['clean_text'])]
    X_train = tf.stack(X_train, axis=0) 
    X_valid = tf.stack(X_valid, axis=0)

    y_train, _ = tocat_encode_labels(df_train['label'])
    y_valid, _ = tocat_encode_labels(df_valid['label'])

    vocab_size = len(vectorize_layer.get_vocabulary()) + 1

    model = build_model(vocab_size = vocab_size,
                        embedding_dim = embedding_vector_size,
                        maxlen = maxlen,
                        neurons = [64, 32, 16, 1])

    history = train_neural_network(X_train = X_train, y_train = y_train,
                                   X_valid = X_valid, y_valid = y_valid,
                                   model = model, batch_size = batch_size,
                                   epochs = epochs, learning_rate = learning_rate,
                                   checkpoint_path = checkpoint_path
                                    )

    write_history(history = history, checkpoint_path = checkpoint_path)                        




if __name__ == '__main__':
    main()
