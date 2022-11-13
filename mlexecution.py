from read_write_data import read_data
from vectorize_data import get_vocabulary, vectorize_X_y_data
from prediction_results import prediction, visualize_results
import configparser
import sys
import pandas as pd

from gensim.models import Word2Vec

#for model-building
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm


def train_classificator(clf, X_train, y_train):
    clf.fit(X_train, y_train)
    return clf


def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    input_folder = config_parse.get('INPUT_OUTPUT', 'folder_preprocessed_datasets')

    df_train, df_valid, df_test = read_data(input_folder=input_folder)
    embedding_vector_size = int(config_parse.get('PARAMETERS_TRAIN', 'embedding_vector_size'))
    vocabulary = get_vocabulary((df_train['clean_text'], df_valid['clean_text'], df_test['clean_text']))
    
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5, min_count=2, workers=4)   

    keyed_vectors = modelw2v.wv  # structure holding the result of training
    # weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array    
    # index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?
    print("The three most common words:")
    for word in keyed_vectors.index_to_key[:3]:
        print(word)

    df_train_val = pd.concat([df_train, df_valid], ignore_index = True)

    X_train, y_train = vectorize_X_y_data((df_train_val['clean_text'], df_train_val['label']), modelw2v)
    #X_val, y_val = vectorize_X_y_data((df_valid['clean_text'], df_valid['label']), modelw2v)
    X_test, y_test = vectorize_X_y_data((df_test['clean_text'], df_test['label']), modelw2v)

    print("Logistic Regression:")
    lr_w2v = train_classificator(LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2'), X_train, y_train)
    y_predict, y_prob = prediction(lr_w2v, X_test)
    visualize_results(y_test, y_predict, y_prob)
    print("Accuracy: {}".format(lr_w2v.score(X_test, y_test)))

    print("\nDecision Tree Classifier:")
    dec_tree_w2v = train_classificator(DecisionTreeClassifier(), X_train, y_train)
    y_predict, y_prob = prediction(dec_tree_w2v, X_test)
    visualize_results(y_test, y_predict, y_prob)
    print("Accuracy: {}".format(dec_tree_w2v.score(X_test, y_test)))

    print("\nSupport Vector Machine:")
    svm_w2v = train_classificator(svm.SVC(probability = True), X_train, y_train)
    y_predict, y_prob = prediction(svm_w2v, X_test)
    visualize_results(y_test, y_predict, y_prob)
    print("Accuracy: {}".format(svm_w2v.score(X_test, y_test)))
    
    
if __name__ == '__main__':
    main()
