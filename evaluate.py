import pickle
import configparser
import sys
from pathlib import Path
from vectorize_data import get_vocabulary, vectorize_X_y_data, tocat_encode_labels
from read_write_data import read_data
from train import build_model, tensorflow_tokenizer, from_text_to_X_vector
from gensim.models import Word2Vec
import numpy as np
from prediction_results import prediction, visualize_results

def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('INPUT_OUTPUT', 'analysis')
    dataset_folder = 'preprocessed_datasets'

    data = read_data(dataset_folder, analysis_folder)

    embedding_vector_size = int(config_parse.get('PARAMETERS_TRAIN', 'embedding_vector_size'))
    min_count_words_w2v = int(config_parse.get('PARAMETERS_TRAIN', 'min_count_words_w2v'))
    random_state = int(config_parse.get('PREPROCESS', 'seed'))

    checkpoint_path = Path('checkpoint')

    # HERE FOR LOGISTIC REGRESSION
    file_path_lr = checkpoint_path / analysis_folder / 'logistic_regression.sav'
    with open(file_path_lr, 'rb') as file:
        lr_w2v = pickle.load(file)

    df_train, df_valid, df_test = data
    vocabulary = get_vocabulary((df_train['clean_text'], df_valid['clean_text'], df_test['clean_text']))
    
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5, min_count=min_count_words_w2v, workers=1, seed = random_state)   

    keyed_vectors = modelw2v.wv  # structure holding the result of training
    print("The three most common words:")
    for word in keyed_vectors.index_to_key[:3]:
        print(word)

    X_test  = vectorize_X_y_data(df_test['clean_text'], modelw2v)
    y_test, classes  = tocat_encode_labels(df_test['label'])

    y_predict, y_prob = prediction(lr_w2v, X_test)
    visualize_results(y_test, y_predict, y_prob, list(classes),
                      title = 'Logistic regressor confusion matrix')

    result = lr_w2v.score(X_test, y_test)
    print("Logistic regressor model, accuracy: {:5.2f}%".format(100 * result))


    # NEURAL NETWORK FROM HERE
    checkpoint_path_nn = checkpoint_path / analysis_folder / 'best_model.hdf5'
    
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

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.load_weights(checkpoint_path_nn)

    X_test = from_text_to_X_vector(df_test['clean_text'], tokenizer, maxlen)
    y_test, classes = tocat_encode_labels(df_test['label'])

    # evaluate the model
    
    y_predict, y_prob = prediction(model, X_test, neural_network = True)
    visualize_results(y_test, y_predict, y_prob, list(classes),
                      title = 'Bidirection LSTM confusion matrix')

    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("LSTM model, accuracy: {:5.2f}%".format(100 * acc))
    
if __name__ == '__main__':
    main()


