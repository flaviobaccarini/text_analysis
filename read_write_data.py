'''
READ AND WRITE MODULE
======================
This module is dedicated for the reading of the input data files 
and the writing of the processed data after the preproccessing step.
'''
from pathlib import Path
import pandas as pd

def read_data(dataset_path, input_folder: str) -> tuple[pd.DataFrame]:
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
    '''
    path_dataset = Path(dataset_path)
    folder_to_read = path_dataset / input_folder
    csv_paths = list(folder_to_read.glob('**/*.csv'))
    csv_paths_stem = [str(path.stem + path.suffix) for path in csv_paths]
    
    if len(csv_paths_stem) == 1:
        complete_dataframe = pd.read_csv(folder_to_read / csv_paths_stem[0])
        return tuple(complete_dataframe)
    elif len(csv_paths_stem) == 2:
        train_dataframe, test_dataframe = read_two_dataframes(folder_to_read, csv_paths_stem)
        return train_dataframe, test_dataframe
    else:
        train_dataframe, valid_dataframe, test_dataframe = read_three_dataframes(folder_to_read, csv_paths_stem)
        return train_dataframe, valid_dataframe, test_dataframe


def read_three_dataframes(datasets_path, csv_paths_stem):
    path_dict = {'train': None, 'val': None, 'test': None}
    for phase in path_dict:
        path_dict[phase] = [csv_path for csv_path in csv_paths_stem if phase in str(csv_path).lower()]
        if len(path_dict[phase]) > 1:
            path_dict[phase] = handle_multiple_occurencies(path_dict[phase], phase)
        else:
            path_dict[phase] = path_dict[phase][0]

    train_ds = pd.read_csv(datasets_path / path_dict['train'], index_col=False)
    val_ds = pd.read_csv(datasets_path / path_dict['val'], index_col=False)
    test_ds = pd.read_csv(datasets_path / path_dict['test'], index_col=False)
    return train_ds, val_ds, test_ds

def read_two_dataframes(datasets_path, csv_paths_stem):
    path_dict = {'train': None, 'test': None}
    for phase in path_dict:
        path_dict[phase] = [csv_path for csv_path in csv_paths_stem if phase in str(csv_path).lower()]
        if len(path_dict[phase]) > 1:
            path_dict[phase] = handle_multiple_occurencies(path_dict[phase], phase)
        else:
            path_dict[phase] = path_dict[phase][0]

    train_ds = pd.read_csv(datasets_path / path_dict['train'], index_col=False)
    test_ds = pd.read_csv(datasets_path / path_dict['test'], index_col=False)
    return train_ds, test_ds

def handle_multiple_occurencies(paths_list, word_to_count):
    paths_dict = dict.fromkeys(paths_list, None)
    for index, key in enumerate(paths_dict):
        paths_dict[key] = paths_list[index].count(word_to_count)

    return max(paths_dict, key = paths_dict.get)

def split_dataframe(dataframes_list, fractions):
    train_frac, val_frac, test_frac = fractions
    if len(dataframes_list) == 2:
        df_train = dataframes_list[0].sample(frac = train_frac)
        df_valid = dataframes_list[0].drop(df_train.index)
        df_test = dataframes_list[1]
        return df_train, df_valid, df_test
    else:
        df_test = dataframes_list[0].sample(frac = test_frac)
        df_train = dataframes_list[0].drop(df_test.index)
        df_valid = df_train.sample(n = int(val_frac*len(dataframes_list[0])))
        df_train = df_train.drop(df_valid.index)
        return df_train, df_valid, df_test

def clean_dataframe(dfs_raw, column_names):
    df_raw_train, df_raw_val, df_raw_test = dfs_raw
    correct_dataframes = []
    for dataframe in (df_raw_train, df_raw_val, df_raw_test):
        df_new_correct = dataframe[[column_names[0], column_names[1]]] # COLUMN NUMBER 0: TEXT, COLUMN NUMBER 1: LABEL
        df_new_correct = df_new_correct.rename({column_names[0]: 'text'}, axis = 'columns')
        correct_dataframes.append(df_new_correct)
    return correct_dataframes

def write_data(dataframes: tuple[pd.DataFrame], output_folder: str) -> None:
    '''
    This function writes the preprocessed data in csv files, divided between 
    train, validation and test, in the output folder passed as parameter.
    The name of the new files, containing the preprocessed dataframes, is the same 
    as the previous files, containing the original data, with a 'mod' sigle at the end of
    the name.

    Parameters:
    ===========
    dataframes: tuple[pandas.DataFrame] or list[pands.DataFrame]
                This tuple contains all the preprocessed dataframes (train, validation, test)
                ready to be stored in a file.

    output_folder: string
                  The output folder path or the output folder name if the folder
                  is inside this current folder.
    '''
    datasets_path = Path('preprocessed_datasets')
    datasets_path = datasets_path / output_folder
    datasets_path.mkdir(parents=True, exist_ok=True)
    df_train, df_val, df_test = dataframes
    df_train.to_csv(path_or_buf = datasets_path / f'{output_folder}_train_preprocessed.csv', index=False)
    df_val.to_csv(path_or_buf = datasets_path / f'{output_folder}_val_preprocessed.csv', index=False)
    df_test.to_csv(path_or_buf = datasets_path / f'{output_folder}_test_preprocessed.csv', index=False)

