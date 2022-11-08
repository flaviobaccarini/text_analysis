'''
READ AND WRITE MODULE
======================
This module is dedicated for the reading of the input data files 
and the writing of the processed data after the preproccessing step.
'''
from pathlib import Path
import pandas as pd


def read_data(input_folder: str) -> tuple[pd.DataFrame]:
    '''
    This function reads the data in csv files, divided between 
    train, validation and test, from the input folder given as parameter to this function.
    The files have to be in this format: three different files, so data must be already
    split in train, validation and test datasets. The train dataset file must contain
    the string "train", while the validation dataset must contain the string "val"
    and the test dataset has to contain "test". These three words are not case sensitive.
    All the file names have to be separated by underscores "_".

    Parameters:
    ===========
    input_folder: string
                  The input folder path or the input folder name if the folder
                  is inside this current folder where the three different files with data
                  are stored.

    Returns:
    =========
    train_ds: pandas.DataFrame
              The train dataframe.

    val_ds:   pandas.DataFrame
              The validation dataframe.
    
    test_ds:  pandas.DataFrame
              The test dataframe.
    '''
    datasets_path = Path(input_folder)
    csv_paths = list(datasets_path.glob('**/*.csv'))
    csv_paths_stem = [str(path.stem + path.suffix) for path in csv_paths]
    
    path_dict = {'train': None, 'val': None, 'test': None}
    for phase in path_dict:
        # i want to eliminate the first word: "constrain" which contain the word "train"
        # also lower all the characters in order to have not a case sensitive problem
        path_dict[phase] = [csv_path for csv_path in csv_paths_stem if phase in "".join(str(csv_path).split("_")[1:]).lower()][0]
    train_ds = pd.read_csv(datasets_path / path_dict['train'])
    val_ds = pd.read_csv(datasets_path / path_dict['val'])
    test_ds = pd.read_csv(datasets_path / path_dict['test'])
    return train_ds, val_ds, test_ds

def write_data(dataframes, output_folder):
    '''
    This function writes the preprocessed data in csv files, divided between 
    train, validation and test, in the output folder passed as parameter.
    The name of the new files, containing the preprocessed dataframes, is the same 
    as the previous files, containing the original data, with a 'mod' sigle at the end of
    the name.

    Parameters:
    ===========
    dataframes: tuple(pandas.DataFrame)
                This tuple contains all the preprocessed dataframes (train, validation, test)
                ready to be stored in a file.

    output_folder: string
                  The output folder path or the output folder name if the folder
                  is inside this current folder.
    '''
    datasets_path = Path(output_folder)
    datasets_path.mkdir(parents=True, exist_ok=True)
    df_train, df_val, df_test = dataframes
    df_train.to_csv(path_or_buf = datasets_path / 'Constraint_English_Train_mod.csv')
    df_val.to_csv(path_or_buf = datasets_path / 'Constraint_English_Val_mod.csv')
    df_test.to_csv(path_or_buf = datasets_path / 'english_test_with_labels_mod.csv')


prova = read_data('datasets')