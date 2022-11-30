'''
Test functions for the reading and writing module.
'''
from binary_classifier.read_write_data import read_data, write_data
from binary_classifier.read_write_data import read_three_dataframes
from binary_classifier.read_write_data import read_two_dataframes
from binary_classifier.read_write_data import handle_multiple_occurencies
import pandas as pd
from pathlib import Path
import string    
import random
from hypothesis import strategies as st
from hypothesis import given
import unittest

def create_fake_data(phases: list[str], 
                    nr_of_rows_per_phase:list[int] = None) -> list[pd.DataFrame]:
    '''
    Helper function for the creation of fake data

    Parameters:
    ===========
    phases: list[str]
            This sequence represents the phases for the data (train/val/test).
    
    nr_of_rows_per_phase: list[int]
                          This sequence represents how many rows for each phase.
                          This sequence has to have the same length of phases.

    Returns:
    ========
    df_fakes: list[pd.DataFrame]
              Fake generated pandas Dataframe.
    '''

    df_fakes = [] 
    # if number of rows per phase is not defined
    # the number of rows per each phase is equal to 100
    if nr_of_rows_per_phase is None:
        nr_of_rows_per_phase = [100 for _ in range(len(phases))]
    for phase, nr_rows in  zip(phases, nr_of_rows_per_phase): 
            fake_data = ({'phase': [phase for _ in range(nr_rows)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase
                                                 + string.digits, k = 10)) 
                                                 for _ in range(nr_rows)], 
                         'id': [int(index) for index in range(nr_rows)], 
                         'label': [random.choice(['real', 'fake']) 
                                                 for _ in range(nr_rows)]}) 
            fake_data_dataframe = pd.DataFrame(fake_data)
            df_fakes.append(fake_data_dataframe) 
    return df_fakes

def test_read_data():
    '''
    This function tests the read_data behaviour.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.
    '''
    # create fake folder
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    # create fake data
    phases = ('train', 'val', 'test')
    df_fake_train, df_fake_val, df_fake_test = create_fake_data(phases)

    # write data inside the folder
    csv_path_stem = ['fake_train_dataset.csv',
                    'fake_val_dataset.csv',
                    'fake_test_dataset.csv']
    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    # read the data
    dfs_read = read_data(new_test_folder)

    assert(len(dfs_read) == 3) # contains 3 different dfs

    df_train, df_val, df_test = dfs_read
    assert(df_train['phase'][0] == 'train') # the dataframe "train" is the one for train
    assert(df_val['phase'][0] == 'val') # the dataframe "val" is the one for validation
    assert(df_test['phase'][0] == 'test') # the dataframe "test" is the one for test

    # 4 columns: phase, tweet, id, label
    assert(len(df_train.columns) == 4) 
    assert(len(df_val.columns) == 4)
    assert(len(df_test.columns) == 4)   

    assert(df_train.columns[0] == 'phase') # the first column is phase 
    assert(df_val.columns[1] == 'tweet') # the second column is the text
    assert(df_val.columns[2] == 'id') # the third column is the id
    assert(df_test.columns[3] == 'label') # the fourth column is the label

    # ELIMINATE THE CREATED FILES AND THE FOLDER
    for csv_path in csv_path_stem:   
        (new_test_folder / csv_path).unlink()
 
    new_test_folder.rmdir()

def test_read_data_capital_letters():
    '''
    This function tests the correct reading of the data (read_data function).
    In particular, in this test function the correct reading indipendent
    from the presence of capital letters in the file names is tested.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.
    '''
    # create fake folder
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    # create fake data
    phase = ('train', 'val', 'test')
    df_fake_train, df_fake_val, df_fake_test = create_fake_data(phase)

    # write fake data in the fake folder
    csv_path_stem = ['fake Train dataset.csv',
                    'fake_Val_dataset.csv',
                    'fake_Test_dataset.csv']
    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    # read the data
    dfs_read = read_data(new_test_folder)

    assert(len(dfs_read) == 3) # contains 3 different dfs

    df_train, df_val, df_test = dfs_read
    assert(df_train['phase'][0] == 'train') # the dataframe "train" is the one for train
    assert(df_val['phase'][0] == 'val') # the dataframe "val" is the one for validation
    assert(df_test['phase'][0] == 'test') # the dataframe "test" is the one for test

    # ELIMINATE THE CREATED FILES AND THE FOLDER
    for csv_path in csv_path_stem:   
        (new_test_folder / csv_path).unlink()
    
    # rewrite the data with different names inside the folder
    csv_path_stem = ['FaKe trAin_dataset.csv',
                    'Fake val Dataset.csv',
                    'Fake_Test_Dataset.csv']
    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    # read the data
    dfs_read = read_data(new_test_folder)

    assert(len(dfs_read) == 3) # contains 3 different dfs

    df_train, df_val, df_test = dfs_read
    assert(df_train['phase'][0] == 'train') # the dataframe "train" is the one for train
    assert(df_val['phase'][0] == 'val') # the dataframe "val" is the one for validation
    assert(df_test['phase'][0] == 'test') # the dataframe "test" is the one for test

    # ELIMINATE THE CREATED FILES AND THE FOLDER
    for csv_path in csv_path_stem:   
        (new_test_folder / csv_path).unlink()
    
    new_test_folder.rmdir()


def test_read_three_df():
    '''
    This function tests the read_three_dataframe function.
    The dataset are already split in three different csv files, ready
    to be read from read_three_dataframe function.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.
    '''
    # create fake folder
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    # create fake data
    phase = ('train', 'val', 'test')
    df_fake_train, df_fake_val, df_fake_test = create_fake_data(phase)

    # write the created data in the folder
    csv_path_stem = ['fake_train_dataset.csv',
                    'fake_val_dataset.csv',
                    'fake_test_dataset.csv']
    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    # read the data
    dfs_read = read_three_dataframes(new_test_folder, csv_path_stem)

    assert(len(dfs_read) == 3) # contains 3 different dfs

    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()
    new_test_folder.rmdir()

@given(name_phase = st.sampled_from(('val', 'test', 'random_name_phase')))
def test_read_twodf(name_phase):
    '''
    This function tests the behaviour of read_two_dataframes function.
    Data are already split in two datasets (train dataset and valid/test dataset).
    Regarding the train dataset: in the filename there must be a 'train' word.
    Instead, regarding the valid/test dataset no requests are needed.
    The name of this dataset, actually, could be everything.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.

    @given:
    =======
    name_phase: str
                Name of the phase that we are considering (no train).
                It could be 'val', 'test', 'random_name_phase'.
    '''
    # create fake folder
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    # create fake data
    # train is always requested, the other could be everything
    phases = ('train', name_phase)
    df_fake_train, df_fake_val = create_fake_data(phases)

    # write data into the fake folder
    # in one file name there must be 'train'
    # no request on the other file name (here try with 'val', 'test', 'random_name_phase)
    csv_path_stem = ['fake_train_dataset.csv',
                    'fake_' + name_phase +'_dataset.csv',]
    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    
    # read data
    dfs_read = read_two_dataframes(new_test_folder, csv_path_stem)

    assert(len(dfs_read) == 2) # contains 2 different dfs

    # test if the dataset are correctly read
    df_train, df_valid = dfs_read

    for dataframe, phase in zip([df_train, df_valid], ['train', name_phase]):
        # check that each dataframe has the same phase word
        # for df_train 'train', for df_valid 'valid', for df_test 'test'
        # for the whole column
        assert((dataframe['phase'] == dataframe['phase'][0]).all())
        # check that in df_train the phase is 'train'
        # df_valid 'valid' and df_test 'test'
        assert((dataframe['phase'] == phase).all()) 
    
    # remove the data and eliminate the folder
    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()
    new_test_folder.rmdir()


@given(number_of_rows = st.integers(min_value = 0, max_value = 100))
def test_read_singledf(number_of_rows):
    '''
    This function tests the working of read_data, if the data 
    are given in a single csv file.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.

    @given:
    ========
    number_of_rows: int
                    Number of rows for the generated csv file.
    '''
    # create fake folder
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    # create fake data
    phases = ('single_csv')
    df_complete = create_fake_data(phases, (number_of_rows,))[0]

    # write the csv into the created folder
    csv_path_stem = ['Dataset_With_Random_Name.csv']
    df_complete.to_csv(new_test_folder / csv_path_stem[0], index=False)

    # read the data
    dfs_fake = read_data(new_test_folder)
    assert(len(dfs_fake) == 1) # contains one single dataframe

    # remove the files and eliminate the folder
    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()
    new_test_folder.rmdir()

def test_read_4df():
    '''
    Test function to test the behaviour of the read_data function
    if inside the input folder there are more than three different files.

    If inside the input folder there are more than three files
    a ValueError is raised.
    '''
    # create fake folder
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    # create fake data
    phases = ('train', 'val', 'test', 'anotherone')
    df_fake_train, df_fake_val, df_fake_test, df_newone = create_fake_data(phases)

    # write data into the created folder
    csv_path_stem = ['fake_train_dataset.csv',
                    'fake_val_dataset.csv',
                    'fake_test_dataset.csv',
                    'fakeboh_test_dataset.csv']
    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)
    df_newone.to_csv(new_test_folder / csv_path_stem[3], index=False)

    # we expect that a ValueError is raised:
    with unittest.TestCase.assertRaises(unittest.TestCase, 
                                        expected_exception = ValueError):
        dfs_read = read_data(new_test_folder)

    # remove the created files and folder
    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()    
    new_test_folder.rmdir()


def test_read_3df_multiple_word():
    '''
    This function tests the behaviour of read_three_dataframes.
    In particular, this test function prove the correct reading of the data
    also if in the analysis name there is a 'train' word.
    Example: if a file name is "constrain_english_train.csv" inside "constrain"
    word there is a "train" word. This could be an issue for the read_three_dataframes
    function, because the function search inside each file names the words "train", "val"
    and "test". When the function finds a match it initializes a pandas dataframe with 
    the content of the matched csv file. Obviously this works well if there is a single
    match between csv files and each word.
    In the case of "constrain_english_train.csv" it would match not only the train file
    to the train dataset, but also the validation and test files (assuming that the
    validation and test file names are: "constrain_englis_validation.csv" and 
    "constrain_english_test.csv").
    Actually, this effect doesn't happen and the function can recognize well which is 
    the real train, validation and test dataset. 
    '''
    # create fake folder to storage the data
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    # create fake data
    phases = ('train', 'val', 'test')
    df_fake_train, df_fake_val, df_fake_test = create_fake_data(phases)

    # try if in the analysis name there is a "train" word
    csv_path_stem = ['constrain_train_dataset.csv',
                    'constrain_val_dataset.csv',
                    'constrain_test_dataset.csv']

    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    # read data
    dfs_read = read_three_dataframes(new_test_folder, csv_path_stem)

    assert(len(dfs_read) == 3) # contains 3 different dfs

    # test if the dataset are correctly read
    df_train, df_valid, df_test = dfs_read

    for dataframe, phase in zip([df_train, df_valid, df_test], phases):
        # check that each dataframe has the same phase word
        # for df_train 'train', for df_valid 'valid', for df_test 'test'
        # for the whole column
        assert((dataframe['phase'] == dataframe['phase'][0]).all())
        # check that in df_train the phase is 'train',
        # df_valid 'valid' and df_test 'test'
        assert((dataframe['phase'] == phase).all()) 

    # remove data and eliminate the folder
    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()
    new_test_folder.rmdir()


@given(phase = st.sampled_from(('train', 'val', 'test')))
def test_multiple_occurencies(phase):
    '''
    This function tests the behaviour of handle_multiple_occurencies function.
    handle_multiple_occurencies is the function that correctly matches the dataframes
    to their corresponding csv files.
    The reading functions (read_three_dataframes or read_two_dataframes) work searching
    some "special" words inside the file names in order to convert the csv files content
    into a pandas dataframe. The special words are:
    for the read_three_dataframes:
        "train", "val", "test"
    for the read_two_dataframes:
        "train"
    Example: "random_train.csv", "random_valid.csv", "random_test.csv", the
    read_three_dataframes matches the "random_train.csv" as the train dataset,
    "random_valid.csv" as the validation dataset and the "random_test.csv" as the 
    test dataset.
    In another case, for example "english_constrain_train.csv", "english_constrain_val.csv"
    and "english_constrain_test.csv" the match is not easy, because of the inside 
    "constrain" there is a "train". In this way we need to take care of this issue
    and let the function know which is the real train dataset.
    handle_multiple_occurencies search inside all the possible matches which is the 
    file name that contains the higher number of "train" (in this case, but it can
    also be "val" or "test"). We expect, in fact, that the train file name contain
    a "train" word more than the other file names.

    @given:
    ========
    phase: str
           It represents the phase of the data.
           The value could be: "train", "val", "test"
    '''
    # the file names would be something like:
    # "trywithtrain_train.csv", "trywithtrain_val.csv" 
    # "triwithtrain_test.csv" 

    # initialize the name with maximum number of count
    name_max_count = 'trywith' + phase + '_' + phase + '.csv'

    # name of the files
    list_names = ['trywith' + phase + '_train.csv',
                 'trywith' + phase + '_val.csv',
                 'trywith' + phase +  '_test.csv']
    
    word_to_count = phase
    name_with_max_count = handle_multiple_occurencies(list_names, word_to_count)

    # verify that the name with maximum count chosen at the beginning
    # is the same as the word the function finds
    assert(name_with_max_count == name_max_count)

    # now try with some whitespaces in the file names
    name_max_count = 'trywith ' + phase + ' ' + phase + '.csv'

    # name of the files
    list_names = ['trywith ' + phase + ' train.csv',
                 'trywith ' + phase + ' val.csv',
                 'trywith ' + phase +  ' test.csv']

    word_to_count = phase
    name_with_max_count = handle_multiple_occurencies(list_names, word_to_count)

    # verify that the name with maximum count chosen at the beginning
    # is the same as the name the function finds
    assert(name_with_max_count == name_max_count)


@given(phase = st.sampled_from(('Train', 'Val', 'Test')))
def test_multiple_occurencies_capital_letters(phase):
    '''
    This function tests the behaviour of handle_multiple_occurencies function.
    handle_multiple_occurencies is the function that correctly matches the dataframes
    to their corresponding csv files.
    The reading functions (read_three_dataframes or read_two_dataframes) work searching
    some "special" words inside the file names in order to convert the csv files content
    into a pandas dataframe. The special words are:
    for the read_three_dataframes:
        "train", "val", "test"
    for the read_two_dataframes:
        "train"
    Example: "random_train.csv", "random_valid.csv", "random_test.csv", the
    read_three_dataframes matches the "random_train.csv" as the train dataset,
    "random_valid.csv" as the validation dataset and the "random_test.csv" as the 
    test dataset.
    In another case, for example "english_constrain_train.csv", "english_constrain_val.csv"
    and "english_constrain_test.csv" the match is not easy, because of the inside 
    "constrain" there is a "train". In this way we need to take care of this issue
    and let the function know which is the real train dataset.
    handle_multiple_occurencies search inside all the possible matches which is the 
    file name that contains the higher number of "train" (in this case, but it can
    also be "val" or "test"). We expect, in fact, that the train file name contain
    a "train" word more than the other file names.

    This test function tests in particular the behaviour of the 
    hadle_multiple_occurencies with capital letters inside the file names.

    @given:
    ========
    phase: str
           It represents the phase of the data.
           The value could be: "Train", "Val", "Test"

    '''
    # the file names would be something like:
    # "trywithtrain_Train.csv", "trywithtrain_Val.csv",
    # "triwithtrain_Test.csv" 

    # initialize the name with maximum number of count
    name_max_count = 'trywith' + phase.lower() + '_' + phase + '.csv'

    # file names
    list_names = ['trywith' + phase.lower() + '_Train.csv',
                 'trywith' + phase.lower() + '_Val.csv',
                 'trywith' + phase.lower() +  '_Test.csv']
    
    word_to_count = phase.lower()
    name_with_max_count = handle_multiple_occurencies(list_names, word_to_count)

    # verify that the name with maximum count chosen at the beginning
    # is the same as the name the function finds
    assert(name_with_max_count == name_max_count)


def test_write_data():
    '''
    This function tests the behaviour of write_data function.
    The list of dataframes to pass to write_data has to be of legth of 3
    and ordered in this way: train dataframe, validation dataframe, test dataframe.
    In order to do so we will create a "fake" folder and we will write 
    some fake data inside this folder.
    '''
    # create some fake data
    phases = ('train', 'val', 'test')
    df_fakes_list = create_fake_data(phases) 
    
    # create fake folder for writing
    name_folder = 'test_writing'
    path_output = Path(name_folder)
    path_output.mkdir(parents = True, exist_ok = True)
    analysis = 'test_write_function'

    # write the data into the folder
    write_data(df_fakes_list, name_folder, analysis)

    csv_paths = list(path_output.glob('**/*.csv'))

    # writing function writes the three different csv files containing the datasets
    assert(len(csv_paths) == 3) 

    df_train, df_val, df_test = read_data(path_output)
    assert(df_train['phase'][0] == 'train') # the dataframe "train" is the one for train
    assert(df_val['phase'][0] == 'val') # the dataframe "val" is the one for validation
    assert(df_test['phase'][0] == 'test') # the dataframe "test" is the one for test

    # remove files and folder
    for csv_path in csv_paths:
        csv_path.unlink()    
    path_output.rmdir()



