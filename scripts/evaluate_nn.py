'''
BIDIRECTIONAL LSTM TEST SCRIPT
============================
Script for testing the trained Bidirectional LSTM for binary text classification.
'''
import configparser
import sys
from pathlib import Path
from text_analysis.read_write_data import read_data
from text_analysis.vectorize_data import get_vocabulary, flatten_unique_voc, tocat_encode_labels
from text_analysis.vectorize_data import init_vector_layer, vectorize_X_data_tf, calculate_max_len
from text_analysis.results import visualize_results
import pandas as pd
from train_nn import build_model


def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('ANALYSIS', 'folder_name')
    dataset_folder = Path('preprocessed_datasets') / analysis_folder
    embedding_vector_size = int(config_parse.get('PARAMETERS_TRAIN', 'embedding_vector_size'))
   
    # READ THE DATA
    data = read_data(dataset_folder)
    df_train, df_valid, df_test = data

    # PATH FOR PLOTS AND MODEL
    plot_path = Path('plots') / analysis_folder
    plot_path.mkdir(parents = True, exist_ok = True)
    checkpoint_path = Path('checkpoint') / analysis_folder
    checkpoint_path_weights_nn = checkpoint_path / 'best_model.hdf5'
    history_path = checkpoint_path / 'history.csv'
    
    # GET THE INITIAL INPUT SEQUENCE LENGTH
    maxlen = calculate_max_len(df_train['clean_text'])
    # GET THE UNIQUE VOCABULARY 
    text_for_vocabulary = list(df_train['clean_text']) + list(df_valid['clean_text'])
    vocabulary = get_vocabulary(text_for_vocabulary)
    unique_voc = flatten_unique_voc(vocabulary)

    # TEXT VECTORIZATION:
    vectorize_layer = init_vector_layer(maxlen, unique_voc)
    X_test = vectorize_X_data_tf(df_test['clean_text'], vectorize_layer)
    y_test, classes  = tocat_encode_labels(df_test['label'], classes = True)

    # BUILD THE MODEL:
    vocab_size = len(vectorize_layer.get_vocabulary())
    model = build_model(vocab_size = vocab_size,
                        embedding_dim = embedding_vector_size,
                        maxlen = maxlen)
    model.summary()

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # LOAD MODEL WEIGHTS:
    model.load_weights(checkpoint_path_weights_nn)

    # PREDICTION AND SCORE
    y_prob = model.predict(X_test)
    y_predict = y_prob
    y_predict[y_prob > 0.5] = 1
    y_predict[y_prob <= 0.5] = 0

    loss, acc = model.evaluate(X_test, y_test, verbose=2)

    # LOAD THE HISTORY 
    history_df = pd.read_csv(history_path, index_col = False)

    # SEE THE RESULTS
    visualize_results(y_test, y_predict, y_prob, list(classes),
                      name_model = 'Bidirection LSTM',
                      metrics = ['accuracy'], 
                      history = history_df, folder_path=plot_path)
    print("LSTM model, accuracy: {:5.2f}%".format(100 * acc))


if __name__ == '__main__':
    main()


