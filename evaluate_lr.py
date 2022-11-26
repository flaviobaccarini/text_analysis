import pickle
import configparser
import sys
from pathlib import Path
from binary_classifier.vectorize_data import vectorize_X_data_lr, tocat_encode_labels
from binary_classifier.read_write_data import read_data
from gensim.models import Word2Vec
from binary_classifier.prediction_results import prediction, visualize_results


def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('INPUT_OUTPUT', 'analysis')
    dataset_folder = Path('preprocessed_datasets') / analysis_folder

    data = read_data(dataset_folder)
    _, _, df_test = data

    plot_path = Path('plots') / analysis_folder
    plot_path.mkdir(parents = True, exist_ok = True)
    checkpoint_path = Path('checkpoint') / analysis_folder

    # HERE FOR LOGISTIC REGRESSION
    file_path_model_w2v = checkpoint_path / 'word2vec.model'
    modelw2v = Word2Vec.load(str(file_path_model_w2v))

    file_path_lr = checkpoint_path / 'logistic_regression.sav'
    with open(file_path_lr, 'rb') as file:
        lr_w2v = pickle.load(file)

    
    X_test  = vectorize_X_data_lr(df_test['clean_text'], modelw2v)
    y_test, classes  = tocat_encode_labels(df_test['label'])

    keyed_vectors = modelw2v.wv  # structure holding the result of training
    print("The three most common words:")
    for word in keyed_vectors.index_to_key[:3]:
        print(word)

    y_predict, y_prob = prediction(lr_w2v, X_test)
    

    acc = lr_w2v.score(X_test, y_test)
    visualize_results(y_test, y_predict, y_prob, list(classes),
                      title = 'Logistic regressor', folder_path=plot_path)
    print("Logistic regressor model, accuracy: {:5.2f}%".format(100 * acc))
    

if __name__ == '__main__':
    main()


