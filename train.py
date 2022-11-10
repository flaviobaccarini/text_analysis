from read_write_data import read_data
from vectorize_data import get_vocabulary, vectorize_X_y_data
import configparser
import sys

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding
#from tensorflow.keras.layers.embeddings import Embeddingà
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

#for model-building
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score
import pandas as pd

def build_model(modelw2v):
    model = Sequential()
    model.add(gensim_to_keras_embedding(modelw2v, False))
    model.add(Bidirectional(LSTM(128, recurrent_dropout=0)))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    return model



def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    input_folder = config_parse.get('INPUT_OUTPUT', 'folder_preprocessed_datasets')

    df_train, df_valid, df_test = read_data(input_folder=input_folder)
    embedding_vector_size = 256
    input_length = 70
    vocabulary = get_vocabulary(df_train['clean_text'], df_valid['clean_text'])
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5, min_count=1, workers=4)   
    #print(modelw2v.wv.index_to_key)
    keyed_vectors = modelw2v.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array    
    index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

    print("The three most common words:")
    for word in keyed_vectors.index_to_key[:3]:
        print(word)


    '''
    model_rnn = build_model(modelw2v)
    model_rnn.compile(
    loss="binary_crossentropy",
    optimizer='adam',
    metrics=['accuracy'])

    history = model_rnn.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=100,
    epochs=20)
    '''
    
if __name__ == '__main__':
    main()
