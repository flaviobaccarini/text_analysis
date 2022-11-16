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
    '''
    if type(input_folder) == str:
        input_folder = Path(input_folder) 

    csv_paths = list(input_folder.glob('**/*.csv'))
    csv_paths_stem = [str(path.stem + path.suffix) for path in csv_paths]
    
    if len(csv_paths_stem) == 1:
        complete_dataframe = pd.read_csv(input_folder / csv_paths_stem[0], index_col=False)
        return (complete_dataframe)
    elif len(csv_paths_stem) == 2:
        train_dataframe, test_dataframe = read_two_dataframes(input_folder, csv_paths_stem)
        return train_dataframe, test_dataframe
    else:
        train_dataframe, valid_dataframe, test_dataframe = read_three_dataframes(input_folder, csv_paths_stem)
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
    
    path_dict['train'] = [csv_path for csv_path in csv_paths_stem if 'train' in str(csv_path).lower()]
    if len(path_dict['train']) > 1:
            path_dict['train'] = handle_multiple_occurencies(path_dict['train'], 'train')
    else:
            path_dict['train'] = path_dict['train'][0]
    test_valid_name = [csv_path for csv_path in csv_paths_stem if not path_dict['train'] in str(csv_path).lower()][0]
    '''for phase in path_dict:
        path_dict[phase] = [csv_path for csv_path in csv_paths_stem if phase in str(csv_path).lower()]
        if len(path_dict[phase]) > 1:
            path_dict[phase] = handle_multiple_occurencies(path_dict[phase], phase)
        else:
            path_dict[phase] = path_dict[phase][0]
    '''
    train_ds = pd.read_csv(datasets_path / path_dict['train'], index_col=False)
    test_ds = pd.read_csv(datasets_path / test_valid_name, index_col=False)
    return train_ds, test_ds

def handle_multiple_occurencies(paths_list, word_to_count):
    paths_dict = dict.fromkeys(paths_list, None)
    for index, key in enumerate(paths_dict):
        paths_dict[key] = paths_list[index].count(word_to_count)

    return max(paths_dict, key = paths_dict.get)

def split_dataframe(dataframes_list, fractions, seed):
    train_frac, test_frac = fractions
    if len(dataframes_list) == 2:
        df_train, df_valid, df_test = split_two_dataframes(dataframes_list, train_frac, seed)
    else:
        df_train, df_valid, df_test = split_single_dataframe(dataframes_list,
                                                             (train_frac, test_frac), seed)

    df_train = df_train.reset_index()
    df_valid = df_valid.reset_index()
    df_test = df_test.reset_index()
    return df_train, df_valid, df_test


def split_single_dataframe(single_dataframe, fractions, seed):
    train_frac, test_frac = fractions
    df_test = single_dataframe.sample(frac = test_frac, random_state = seed)
    df_train_val = single_dataframe.drop(df_test.index)
    df_train = df_train_val.sample(n = int(train_frac*len(single_dataframe)), random_state = seed)
    df_valid = df_train_val.drop(df_train.index)
    return df_train, df_valid, df_test

def split_two_dataframes(dataframes, train_frac, seed):
    n_frac = int( (len(dataframes[0]) + len(dataframes[1])) * train_frac)
    df_train = dataframes[0].sample(n = n_frac, random_state = seed)
    df_valid = dataframes[0].drop(df_train.index)
    df_test = dataframes[1]
    return df_train, df_valid, df_test

#TODO: sposta questa funzione in preprocess e aggiungici una parte per eliminare eventuali righe vuote
def clean_dataframe(dfs_raw, column_names):
    df_raw_train, df_raw_val, df_raw_test = dfs_raw
    correct_dataframes = []

    for dataframe in (df_raw_train, df_raw_val, df_raw_test):
        df_new_correct = dataframe.loc[:, list(column_names)] # COLUMN NUMBER 0: TEXT, COLUMN NUMBER 1: LABEL
        dict_new_names = {column_names[0]: 'text', column_names[1]: 'label'}
        df_new_correct = df_new_correct.rename(dict_new_names, axis = 'columns')
        correct_dataframes.append(df_new_correct)
    
    return correct_dataframes

def write_data(dataframes: tuple[pd.DataFrame], output_folder: str, analysis: str) -> None:
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

    if type(output_folder) == str:
        output_folder = Path(output_folder)
    
    df_train, df_val, df_test = dataframes
    df_train.to_csv(path_or_buf = output_folder / f'{analysis}_train_preprocessed.csv', index=False)
    df_val.to_csv(path_or_buf = output_folder / f'{analysis}_val_preprocessed.csv', index=False)
    df_test.to_csv(path_or_buf = output_folder / f'{analysis}_test_preprocessed.csv', index=False)

