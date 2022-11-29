'''
READING AND WRITING MODULE
===========================
This module is dedicated for the reading of the input data files 
and the writing of the processed data after the preproccessing step.
'''

from pathlib import Path
import pandas as pd
import numpy as np

def read_data(input_folder: str) -> tuple[pd.DataFrame]:
    '''
    This function reads the data in csv files 
    from the input folder given as parameter to this function.

    If inside the folder there are three different files, this
    function assumes that data are already split. For that reason
    the train dataset file must contain the string "train",
    while the validation dataset must contain the string "val"
    and the test dataset has to contain "test".

    If inside the folder there are only two different files, this function
    assumes that data are already split between train and test.
    The train dataset file must contain the "train" string,
    no request are needed for the other file contained in the folder.

    If inside the folder there is only one single, this function
    assumes that data are not alread split.
    No request on the name of the file.

    The strings "train", "val", "test" that have to be in the filenames
    are not case sensitive.

    Parameters:
    ===========
    input_folder: Path-like or str
                  The input folder path to the files.

    Raises:
    ========
    ValueError: if the number of files inside the folder are greater
                than 3.

    Returns:
    =========
                 tuple[pd.DataFrame]
                 The tuple contains the dataframes read from the csv files
                 inside the input folder.
                 If there is only one single csv file, a tuple of length 1 
                 is returned.
                 If there are two different csv files, a tuple of length 2
                 is returned (train and test dataframes).
                 If there are three different csv files, a tuple of length 3
                 is returned (train, validation and test dataframes).
    '''
    if type(input_folder) == str:
        input_folder = Path(input_folder) 

    csv_paths = list(input_folder.glob('**/*.csv'))
    csv_paths_stem = [str(path.stem + path.suffix) for path in csv_paths]
    
    if len(csv_paths_stem) == 1:
        complete_dataframe = pd.read_csv(input_folder / csv_paths_stem[0], index_col=False)
        return complete_dataframe,

    elif len(csv_paths_stem) == 2:
        train_dataframe, test_dataframe = read_two_dataframes(input_folder, csv_paths_stem)
        return train_dataframe, test_dataframe

    elif len(csv_paths_stem) == 3:
        train_dataframe, valid_dataframe, test_dataframe = read_three_dataframes(input_folder, csv_paths_stem)
        return train_dataframe, valid_dataframe, test_dataframe

    else: 
        raise ValueError("Too many files inside the input folder." + 
                        f'\nIn {input_folder} there are {len(csv_paths_stem)} files' +
                        '\nExpected at maximum three (train,validation,test).')

def read_three_dataframes(datasets_path: Path, 
                         csv_paths_stem: list[str]) -> tuple[pd.DataFrame]:
    '''
    Helper function for reading data; case of three different files.

    In this case, inside the folder there are three different files. 
    The train dataset file must contain the string "train",
    while the validation dataset must contain the string "val"
    and the test dataset has to contain "test".

    The words "train", "val", "test" that have to be in the filenames
    are not case sensitive.

    Parameters:
    ===========
    datasets_path: Path-like 
                   The input folder path to the files.
    
    csv_paths_stem: list[str]
                    This sequence contains only the file names.
                    The length of this sequence has to be equal to three.

    Returns:
    =========
    train_ds: pd.DataFrame
              This is the train dataframe.
              The data inside this dataframe are the one
              stored in the train csv file.

    val_ds: pd.DataFrame
            This is the validation dataframe.
            The data inside this dataframe are the one
            stored in the validation csv file.

    test_ds: pd.DataFrame
             This is the test dataframe.
             The data inside this dataframe are the one
             stored in the test csv file.
    '''                     
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

def read_two_dataframes(datasets_path: Path,
                        csv_paths_stem: list[str]):
    '''
    Helper function for reading data; case of two different files.

    In this case, inside the folder there are two different files. 
    The train dataset file must contain the string "train",
    while the other dataset (which the function assumes to be the test
    dataset) has no request at all on the file name.

    The string "train" that has to be in the file name
    is not case sensitive.

    Parameters:
    ===========
    datasets_path: Path-like 
                   The input folder path to the files.
    
    csv_paths_stem: list[str]
                    This sequence contains only the file names.
                    The length of this sequence has to be equal to three.

    Returns:
    =========
    train_ds: pd.DataFrame
              This is the train dataframe.
              The data inside this dataframe are the one
              stored in the train csv file.

    test_ds: pd.DataFrame
             This is the test dataframe.
             The data inside this dataframe are the one
             stored in the not train csv file.
    '''      

    path_dict = {'train': None, 'test': None}
    
    path_dict['train'] = [csv_path for csv_path in csv_paths_stem if 'train' in str(csv_path).lower()]
    if len(path_dict['train']) > 1:
            path_dict['train'] = handle_multiple_occurencies(path_dict['train'], 'train')
    else:
            path_dict['train'] = path_dict['train'][0]

    test_name = [csv_path for csv_path in csv_paths_stem if not path_dict['train'] in str(csv_path).lower()][0]

    train_ds = pd.read_csv(datasets_path / path_dict['train'], index_col=False)
    test_ds = pd.read_csv(datasets_path / test_name, index_col=False)

    return train_ds, test_ds

def handle_multiple_occurencies(paths_list: list[str],
                                word_to_count: str) -> str:
    '''
    Helper function for correctly mapping each single dataframe
    to the correct name of the file.

    An example could be useful to understand the reason why and 
    the behaviour of this function.
    Imagine that we have three different files inside a folder.
    This means that one file is for the training dataset, one for the validation
    dataset and the other one for the test dataset. When the user uses the read_data
    function, this function search inside the file names the words "train", "val"
    and "test" in order to initialize three different dataframes corresponding to
    the three different files. If the names are: "random_train.csv", "random_val.csv"
    and "random_test.csv" the train dataset corresponds to "random_train.csv", the
    validation dataset corresponds to "random_val.csv" and the test dataset corresponds
    to "random_test.csv". 
    But what if inside the file names is there another "train" word (or could be "val"
    "test")? For example, if the file names are: "constrain_train.csv",
    "constrain_val.csv", "constrain_test.csv", the only read_data function may can't
    understand which one is the train dataset (because in each file name there
    is a "train" word). Basically handle_multiple_occurencies helps to understand which
    is the real train dataset, looking for the matched file name that contains the highest
    number of occurencies for the "train" word (we can expect that in the train dataset
    file name there is one "train" word more than the other file names.)
    The example is made with the "train" word, but handle_multiple_occurencies works 
    also with "val"/"validation" and "test".

    Parameters:
    ============
    paths_list: list[str]
                List that contains all the possible suitable
                file names for one dataset.
    
    word_to_count: str
                   This is the word that we want to count in the 
                   file names list (paths_list).
                   It could be "train", "val"/"validation" or "test".
                   The handle_multiple_occurencies
                   is going to search inside all the file names
                   contained in the paths_list the
                   number of occurencies for 
                   the word_to_count.

    Returns:
    =========
    filename_for_word: str
                       It represents the file name inside the 
                       paths_list, in which the word_to_count has the 
                       maximum number of occurencies.
    '''

    paths_dict = dict.fromkeys(paths_list, None)
    for index, key in enumerate(paths_dict):
        paths_dict[key] = paths_list[index].lower().count(word_to_count)

    filename_max_occurencies_word = max(paths_dict, key = paths_dict.get)
    return filename_max_occurencies_word

def write_data(dataframes: tuple[pd.DataFrame],
               output_folder: str,
               analysis: str) -> None:
    '''
    This function writes the preprocessed data in csv files, divided between 
    train, validation and test, in the output folder passed as parameter.
    It is important that dataframes tuple (or list) contains three different 
    dataframes in this order: train, validation and test. 
    The name of the new files will be the analysis name followed by;
    "train_preprocessed.csv" or "val_preprocessed.csv" or "test_preprocessed.csv".

    Parameters:
    ===========
    dataframes: tuple[pandas.DataFrame] 
                This tuple contains all the preprocessed dataframes 
                (train, validation, test) ready to be written in a csv file.

    output_folder: Path-like or str
                   The output folder path.

    analysis: str
              Name of the analysis the user is doing.
              For example, if data regarding COVID-19 are analyzed, the 
              analysis name could be "covid".
    '''

    if type(output_folder) == str:
        output_folder = Path(output_folder)
    
    df_train, df_val, df_test = dataframes
    names = [f'{analysis}_train_preprocessed.csv',
             f'{analysis}_val_preprocessed.csv',
             f'{analysis}_test_preprocessed.csv' ]
    df_train.to_csv(path_or_buf = output_folder / names[0], index=False)
    df_val.to_csv(path_or_buf = output_folder / names[1], index=False)
    df_test.to_csv(path_or_buf = output_folder / names[2], index=False)

