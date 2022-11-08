from read_write_data import read_data
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from preprocess import MeanEmbeddingVectorizer, get_vocabulary, vectorize_data
import pandas as pd
import configparser
import sys
import numpy as np
from preprocess import vectorize_data


def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    input_folder = config_parse.get('INPUT_OUTPUT', 'folder_preprocessed_datasets')

    df_train, df_valid, df_test = read_data(input_folder=input_folder)
    vocabulary = get_vocabulary(df_train['clean_tweet'], df_valid['clean_tweet'])
    X_vector_train, y_vector_train = vectorize_data((df_train['clean_tweet'],
                                                     df_train['label']),
                                                    vocabulary,
                                                    tf_bool = False)


if __name__ == '__main__':
    main()
