'''
READING AND WRITING MODULE
===========================
This module is dedicated for the reading of the input data files 
and the writing of the processed data after the preproccessing step.
'''
from pathlib import Path
import pandas as pd

def get_paths(filenames: list[str]) -> dict:
    '''
    Function for getting the correcting file names match if
    in the input folder there are two or three files. 
    The basic idea of this function is to correctly match
    the csv file name with the pandas dataframe that will 
    be created from the csv. 
    Case with 3 files: each file needs to contain a special string
    ('train', 'val', 'test') and this function matches the 
    correct csv path to a dictionary (for the reading of the csv files).
    Case with 2 files: the train file need to be named with 'train' string
    within the file name. The other file could not contain any special string,
    it will be assumed to be the test dataset.

    Parameters:
    ===========
    filenames: list[str]
               Sequence that contains all the file names from the input folder.

    Returns:
    ========
    path_dict: dict
               Dictionary that maps each key to the corresponding file name.
               The key 'val' could be not initialized if inside the folder there
               are only two files.
    '''
    path_dict = {'train': None, 'val': None, 'test': None}
    filenames_remaining = filenames.copy()

    for phase in path_dict:
        path_dict[phase] = [csv_path for csv_path in filenames if phase in str(csv_path).lower()]
        # if multiple matching: choose the rigth one with handle_multiple_occurencies
        if len(path_dict[phase]) > 1:
            path_dict[phase] = handle_multiple_occurencies(path_dict[phase], phase)
            filenames_remaining.remove(path_dict[phase])
        
        # if only one match: flatten the list
        if len(path_dict[phase]) == 1:
            path_dict[phase] = path_dict[phase][0]
            filenames_remaining.remove(path_dict[phase])

    # if in the filenames_remaining there is one element and 
    # at the beginning there are two file inside the folder:
    #  initialize the path_dict['test'] with this file name
    if (len(filenames_remaining) == 1) and (len(filenames) == 2):
        path_dict['test'] = filenames_remaining[0]

    return path_dict

def remove_none_from_dict(original_dict: dict) -> dict:
    '''
    Function for removing from a dictionary an empty list.
    This is useful in the case when we have only two different
    csv files in a folder. After the execution of the function
    get_paths we will find a dictionary with three keys, but
    one key is useless (it's just an empty list). 
    With this function we are able to remove this useless key.

    Parameters:
    ===========
    original_dict: dict
                   Original dictionary that can contains 
                   keys mapped to empty list ([]).
    
    Returns:
    ========
    no_empty_list_dict: dict
                        The original dictionary without keys
                        mapped to empty list ([]).
    '''
    no_empty_list_dict = {k: v for k, v in original_dict.items() if v}
    return no_empty_list_dict


def read_data(input_folder: str) -> tuple[pd.DataFrame]:
    '''
    This function reads the data in csv files 
    from the input folder path given as parameter to this function.

    If inside the folder there are three different files, this
    function assumes that data are already split. For that reason
    the train dataset file must contain the string "train",
    while the validation dataset must contain the string "val"
    and the test dataset has to contain "test" string.

    If inside the folder there are only two different files, this function
    assumes that data are already split between train and test.
    The train dataset file must contain the "train" string,
    no request are needed for the other file contained in the folder.

    If inside the folder there is only one single file, this function
    assumes that data are not alread split.
    No request on the name of the file.

    The strings "train", "val", "test" that have to be within file names
    are not case sensitive.

    Parameters:
    ===========
    input_folder: Path-like or str
                  The input folder path to the files.

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
    filenames = [str(path.stem + path.suffix) for path in csv_paths]
    
    if len(filenames) == 1:
        complete_dataframe = pd.read_csv(input_folder / filenames[0], index_col=False)
        return complete_dataframe,

    else:
        path_dict = get_paths(filenames)
        correct_path_dict = remove_none_from_dict(path_dict)
        dataframes = []
        for phase, filename in correct_path_dict.items():
            df = pd.read_csv(input_folder / filename)
            dataframes.append(df)

        return tuple(dataframes)

def handle_multiple_occurencies(suitable_names: list[str],
                                word_to_count: str) -> str:
    '''
    Helper function for correctly mapping each single dataframe
    to the correct name of the file.

    An example could be useful to understand the reason why and 
    the behaviour of this function.
    Imagine that we have three different files inside a folder.
    This means that one file is for the training dataset, one for the validation
    dataset and the other one for the test dataset. When the user uses the read_data
    function, this function searches inside the file names the words "train", "val"
    and "test" in order to initialize three different dataframes corresponding to
    the three different files. If the names are: "random_train.csv", "random_val.csv"
    and "random_test.csv", the train dataset corresponds to "random_train.csv", the
    validation dataset corresponds to "random_val.csv" and the test dataset corresponds
    to "random_test.csv". 
    But what if inside the file names is there another "train" word (or could be "val"
    "test")? For example, if the file names are: "constrain_train.csv",
    "constrain_val.csv", "constrain_test.csv", the read_data function may can't
    understand which one is the train dataset (because in each file name there
    is a "train" word). Basically handle_multiple_occurencies helps to understand which
    is the real train dataset, looking for the matched file name that contains the highest
    number of occurencies for the "train" word (we can expect that in the train dataset
    file name there is one "train" word more than the other matched file names.)
    The example is made with the "train" word, but handle_multiple_occurencies works 
    also with "val"/"validation" and "test".

    Parameters:
    ============
    suitable_names: list[str]
                    List that contains all the possible suitable
                    file names for one dataset.
    
    word_to_count: str
                   This is the word that we want to count in the 
                   file names list (suitable_names).
                   It could be "train", "val"/"validation" or "test".
                   The handle_multiple_occurencies
                   is going to search inside all the file names
                   contained in the suitable_names the
                   number of occurencies for 
                   the word_to_count.

    Returns:
    =========
    filename_for_word: str
                       It represents the file name inside the 
                       suitable_names, in which the word_to_count has the 
                       maximum number of occurencies.
    '''

    paths_dict = dict.fromkeys(suitable_names, None)
    for index, key in enumerate(paths_dict):
        paths_dict[key] = suitable_names[index].lower().count(word_to_count)

    filename_max_occurencies_word = max(paths_dict, key = paths_dict.get)
    return filename_max_occurencies_word

def write_data(dataframes: tuple[pd.DataFrame],
               output_folder: str,
               analysis: str) -> None:
    '''
    This function writes the preprocessed data in csv files, divided between 
    train, validation and test, in the output folder passed as parameter.
    It is important that dataframes sequence contains three different 
    dataframes in this order: train, validation and test. 
    The name of the new files will be the analysis string followed by;
    "train_preprocessed.csv" or "val_preprocessed.csv" or "test_preprocessed.csv".

    Parameters:
    ===========
    dataframes: tuple[pandas.DataFrame] 
                This sequence contains all the preprocessed dataframes 
                (train, validation, test) ready to be written in csv files.

    output_folder: Path-like or str
                   The output folder path.

    analysis: str
              Name of the analysis the user is doing.
              For example, if data regarding COVID-19 tweets are analyzed, the 
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

