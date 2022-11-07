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
    This functions reads the data in csv files, divided between 
    train, validation and test, from the input folder given as parameter to this function.

    Parameters:
    ===========
    input_folder: string
                  The input folder path or the input folder name if the folder
                  is inside this current folder.

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
    csv_paths_stem = [x.stem + x.suffix for x in csv_paths]
    csv_paths_stem_sorted = sorted(csv_paths_stem)
    train_ds = pd.read_csv(datasets_path / csv_paths_stem_sorted[0])
    val_ds = pd.read_csv(datasets_path / csv_paths_stem_sorted[1])
    test_ds = pd.read_csv(datasets_path / csv_paths_stem_sorted[2])
    return train_ds, val_ds, test_ds

def write_data(dataframes, output_folder):
    '''
    This functions writes the processed data in csv files, divided between 
    train, validation and test in the output folder passed as parameter.
    The name of the new files, containing the preprocessed dataframes, is the same 
    as the previous files, containing the original data, with a 'mod' sigle at the end.

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