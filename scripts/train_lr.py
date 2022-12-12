'''
LOGISTIC REGRESSOR TRAIN SCRIPT
============================
Script for training a logistic regressor for binary text classification.
'''
from text_analysis.read_write_data import read_data
from text_analysis.vectorize_data import get_vocabulary, tocat_encode_labels, vectorize_X_data_lr
import configparser
import sys
from pathlib import Path
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import pickle


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
    min_count_words_w2v = int(config_parse.get('PARAMETERS_TRAIN', 'min_count_words_w2v'))
    random_state = int(config_parse.get('PREPROCESS', 'seed'))
    checkpoint_path = Path('checkpoint') / analysis_folder

    # GET THE VOCABULARY:
    all_train_words = list(df_train['clean_text']) + list(df_valid['clean_text'])
    vocabulary = get_vocabulary(all_train_words)

    # Word2Vec MODEL INIT:
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5,
                         min_count=min_count_words_w2v, workers=1, seed = random_state)   

    # DATA VECTORIZATION:
    df_train_val = pd.concat([df_train, df_valid], ignore_index = True)
    X_train = vectorize_X_data_lr(df_train_val['clean_text'], modelw2v)
    y_train = tocat_encode_labels(df_train_val['label'])

    # CHECKPOINT PATH:
    checkpoint_path.mkdir(parents = True, exist_ok = True)
    file_path_model_w2v = checkpoint_path / 'word2vec.model'
    file_path_model_lr = checkpoint_path / 'logistic_regression.sav'

    # TRAIN LOGISTIC REGRESSOR
    print("Train logistic regressor...")
    lr_w2v = LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2', max_iter=20)
    lr_w2v.fit(X_train, y_train)
    print("Logistic Regression: training done")

    # SAVING LOGISTIC REGRESSOR AND Word2Vec MODEL
    with open(file_path_model_lr, 'wb') as file:
        pickle.dump(lr_w2v, file)
    modelw2v.save(str(file_path_model_w2v))

if __name__ == '__main__':
    main()
