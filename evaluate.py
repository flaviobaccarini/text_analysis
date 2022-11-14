import pickle
import configparser
import sys
from pathlib import Path
from vectorize_data import get_vocabulary, vectorize_X_y_data, tocat_encode_labels
from read_write_data import read_data
from train import build_model, tensorflow_tokenizer, from_text_to_X_vector, calculate_max_len
from gensim.models import Word2Vec
import numpy as np
from prediction_results import prediction, visualize_results
import pandas as pd


def evaluate_logistic_regression(X_test,
                                 y_test,
                                 modelw2v,
                                 lr_w2v):


    keyed_vectors = modelw2v.wv  # structure holding the result of training
    print("The three most common words:")
    for word in keyed_vectors.index_to_key[:3]:
        print(word)

    y_predict, y_prob = prediction(lr_w2v, X_test)
    

    acc = lr_w2v.score(X_test, y_test)

    return y_predict, y_prob, acc


def evaluate_neural_network(X_test, y_test, 
                            embedding_vector_size,
                            vocab_size,
                            maxlen,
                            checkpoint_path_weight_nn):

    model = build_model(vocab_size = vocab_size,
                        embedding_dim = embedding_vector_size,
                        maxlen = maxlen)

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.load_weights(checkpoint_path_weight_nn)


    # evaluate the model
    
    y_predict, y_prob = prediction(model, X_test, neural_network = True)

    loss, acc = model.evaluate(X_test, y_test, verbose=2)

    return y_predict, y_prob, acc

def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('INPUT_OUTPUT', 'analysis')
    dataset_folder = Path('preprocessed_datasets') / analysis_folder

    data = read_data(dataset_folder)
    _, _, df_test = data

    plot_path = Path('plots') / analysis_folder
    plot_path.mkdir(parents = True, exist_ok = True)
    embedding_vector_size = int(config_parse.get('PARAMETERS_TRAIN', 'embedding_vector_size'))
    checkpoint_path = Path('checkpoint') / analysis_folder

    # HERE FOR LOGISTIC REGRESSION
    file_path_model_w2v = checkpoint_path / 'word2vec.model'
    modelw2v = Word2Vec.load(str(file_path_model_w2v))

    file_path_lr = checkpoint_path / 'logistic_regression.sav'
    with open(file_path_lr, 'rb') as file:
        lr_w2v = pickle.load(file)

    
    X_test  = vectorize_X_y_data(df_test['clean_text'], modelw2v)
    y_test, classes  = tocat_encode_labels(df_test['label'])

    y_predict, y_prob, acc = evaluate_logistic_regression(X_test, y_test, modelw2v, lr_w2v)

    visualize_results(y_test, y_predict, y_prob, list(classes),
                      title = 'Logistic regressor', folder_path=plot_path)
    print("Logistic regressor model, accuracy: {:5.2f}%".format(100 * acc))
    


    # NEURAL NETWORK FROM HERE
    checkpoint_path_weights_nn = checkpoint_path / 'best_model.hdf5'
    
    maxlen = calculate_max_len(data[0]['clean_text'])

    max_num_words = int(len(get_vocabulary((data[0]['clean_text'],
                                            data[1]['clean_text'],
                                            data[2]['clean_text']))) * 1.5)
    tokenizer = tensorflow_tokenizer(max_num_words = max_num_words, text = data[0]['clean_text'])

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    X_test = from_text_to_X_vector(df_test['clean_text'], tokenizer, maxlen)

    y_predict, y_prob, acc = evaluate_neural_network(X_test, y_test, embedding_vector_size, vocab_size,
                            maxlen, checkpoint_path_weights_nn)

    history_path = checkpoint_path / 'history.csv'
    history_df = pd.read_csv(history_path, index_col = False)

    visualize_results(y_test, y_predict, y_prob, list(classes),
                      title = 'Bidirection LSTM', history = history_df, folder_path=plot_path)
    print("LSTM model, accuracy: {:5.2f}%".format(100 * acc))


if __name__ == '__main__':
    main()


