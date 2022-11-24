import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding, BatchNormalization

import pandas as pd

from sklearn.linear_model import LogisticRegression
import pickle


def train_neural_network(X_train, y_train,
                        X_valid, y_valid,
                        model,
                        batch_size,
                        epochs,
                        learning_rate,
                        checkpoint_path):

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # CHECKPOINT CALLBACK
    checkpoint_model_path = checkpoint_path / 'best_model.hdf5' 
    model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_model_path,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

    # EARLY STOP CALLBACK
    early_stop_callback = EarlyStopping(monitor='val_loss', patience=20)

    # TRAIN
    history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_valid, y_valid),
    callbacks = [model_checkpoint_callback, early_stop_callback],
    batch_size=batch_size,
    epochs=epochs)
    
    return history

def write_history(history, checkpoint_path):
    # SAVE THE MODEL AND THE HISTORY in a DATAFRAME

    # convert the history.history dict to a pandas DataFrame:     
    hist_df = pd.DataFrame(history.history) 
    # save to csv: 
    hist_csv_file = checkpoint_path / 'history.csv'

    hist_df.to_csv(hist_csv_file, index = False)
    

def train_logistic_regressor(X_train, y_train,
                            file_path
                            ):

    # TRAIN
    print("Train logistic regressor...")
    lr_w2v = LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2', max_iter=20)
    lr_w2v.fit(X_train, y_train)
    print("Logistic Regression: training done")

    # SAVING MODEL
    with open(file_path, 'wb') as file:
        pickle.dump(lr_w2v, file)

# TODO: CREARE UNA FUNZIONE IN train_models PER L'EMBEDDING
def build_model(vocab_size, embedding_dim, maxlen, neurons):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, 
                        output_dim=embedding_dim, 
                        input_length=maxlen))
    model.add(Bidirectional(LSTM(neurons[0], recurrent_dropout=0)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    for nr_neuron in neurons[1:]:
        model.add(Dense(nr_neuron))
        model.add(Dropout(0.25))
    model.add(Dense(neurons[-1], activation='sigmoid'))
    model.summary()

    return model
