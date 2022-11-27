'''
Test functions for the reading and writing module.
'''

from binary_classifier.read_write_data import read_data, write_data, read_three_dataframes
from binary_classifier.read_write_data import read_two_dataframes, handle_multiple_occurencies
#import numpy as np
import pandas as pd
from pathlib import Path
import string    
import random # define the random module  
from hypothesis import strategies as st
from hypothesis import given
import unittest

# TODO: CREARE UNA FUNZIONE PER LA GENERAZIONE DI FAKE DATA 
# COSÌ CHE OGNI FUNZIONE NON DEBBA GENERARE SEMPRE NUOVI DATI
# MA BASTA CHE CHIAMI LA FUNZIONE CHE LO FA

def test_read_data():
    '''
    This function tests the correct reading of the data.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.
    '''
 
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    df_fakes = [] 
    for phase in  ('train', 'val', 'test'): 
            fake_data = ({'phase': [phase for _ in range(100)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(100)], 
                         'id': [int(index) for index in range(100)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(100)]}) 
            fake_data_dataframe = pd.DataFrame(fake_data)
            df_fakes.append(fake_data_dataframe) 

    df_fake_train, df_fake_val, df_fake_test = df_fakes

    csv_path_stem = ['fake_train_dataset.csv',
                    'fake_val_dataset.csv',
                    'fake_test_dataset.csv']
    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    dfs_fake = read_data(new_test_folder)

    assert(len(dfs_fake) == 3) # contains 3 different dfs

    df_train, df_val, df_test = dfs_fake
    assert(df_train['phase'][0] == 'train') # the dataframe "train" is the one for train
    assert(df_val['phase'][0] == 'val') # the dataframe "val" is the one for validation
    assert(df_test['phase'][0] == 'test') # the dataframe "test" is the one for test

    assert(len(df_train.columns) == 4) # phase, tweet, id, label
    assert(len(df_val.columns) == 4)
    assert(len(df_test.columns) == 4)   

    assert(df_train.columns[0] == 'phase') # the second column is phase 
    assert(df_val.columns[1] == 'tweet') # the third column is the text
    assert(df_test.columns[3] == 'label') # the fifth column is the label

    # ELIMINATE THE CREATED FILES AND THE FOLDER
    for csv_path in csv_path_stem:   
        (new_test_folder / csv_path).unlink()
 

    new_test_folder.rmdir()


def test_read_data_capital_letters():
    '''
    This function tests the correct reading of the data, 
    in order to prove the indipendence from the presence of 
    capital letters in the filenames.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.
    '''
 
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    df_fakes = [] 
    for phase in  ('train', 'val', 'test'): 
            fake_data = ({'phase': [phase for _ in range(100)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(100)], 
                         'id': [int(index) for index in range(100)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(100)]}) 
            fake_data_dataframe = pd.DataFrame(fake_data)
            df_fakes.append(fake_data_dataframe) 

    df_fake_train, df_fake_val, df_fake_test = df_fakes

    csv_path_stem = ['fake Train dataset.csv',
                    'fake_Val_dataset.csv',
                    'fake_Test_dataset.csv']

    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    dfs_fake = read_data(new_test_folder)

    assert(len(dfs_fake) == 3) # contains 3 different dfs

    # ELIMINATE THE CREATED FILES AND THE FOLDER
    for csv_path in csv_path_stem:   
        (new_test_folder / csv_path).unlink()
    

    csv_path_stem = ['FaKe trAin_dataset.csv',
                    'Fake val Dataset.csv',
                    'Fake_Test_Dataset.csv']

    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    dfs_fake = read_data(new_test_folder)

    assert(len(dfs_fake) == 3) # contains 3 different dfs

    # ELIMINATE THE CREATED FILES AND THE FOLDER
    for csv_path in csv_path_stem:   
        (new_test_folder / csv_path).unlink()
    

    new_test_folder.rmdir()



def test_write_data():
    '''
    This function tests the correct writing of the data.
    In order to do so we will create a "fake" folder and we will write 
    some fake text data inside this folder.
    '''
    df_fakes_list = [] 
    for phase in  ('train', 'val', 'test'): 
            fake_data = ({'phase': [phase for _ in range(100)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(100)], 
                         'id': [int(index) for index in range(100)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(100)]}) 
            fake_data_dataframe = pd.DataFrame(fake_data) 
            df_fakes_list.append(fake_data_dataframe) 
    
    df_fakes_tuple = tuple(df_fakes_list)
    name_folder = 'test_writing'
    path_output = Path(name_folder)
    path_output.mkdir(parents = True, exist_ok = True)
    analysis = 'test_write_function'

    write_data(df_fakes_tuple, name_folder, analysis)

    csv_paths = list(path_output.glob('**/*.csv'))
    
    assert(len(csv_paths) == 3) # writing function writes the three different csv files containing the datasets

    for csv_path in csv_paths:
        csv_path.unlink()
    

    write_data(df_fakes_list, name_folder, analysis) # it works with both tuple or list

    csv_paths = list(path_output.glob('**/*.csv'))
    
    assert(len(csv_paths) == 3) # writing function writes the three different csv files containing the datasets

    for csv_path in csv_paths:
        csv_path.unlink()
    
    path_output.rmdir()



def test_read_three_df():
    '''
    This function tests the correct reading of the data, if the data 
    are already split in three dataframes.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.
    '''
 
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    df_fakes = [] 
    for phase in  ('train', 'valid', 'test'): 
            fake_data = ({'phase': [phase for _ in range(100)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(100)], 
                         'id': [int(index) for index in range(100)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(100)]}) 
            fake_data_dataframe = pd.DataFrame(fake_data)
            df_fakes.append(fake_data_dataframe) 

    df_fake_train, df_fake_val, df_fake_test = df_fakes


    csv_path_stem = ['fake_train_dataset.csv',
                    'fake_val_dataset.csv',
                    'fake_test_dataset.csv']

    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    dfs_fake = read_three_dataframes(new_test_folder, csv_path_stem)

    assert(len(dfs_fake) == 3) # contains 3 different dfs

    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()

    new_test_folder.rmdir()


def test_read_3df_multiple_word():
    '''
    This function tests the correct reading of the data, if the data 
    are already split in three dataframes.
    In particular, this test function prove the correct reading of the data
    also if in the analysis name there is a 'train' (example: constrain contain train).
    The example is made with 'constrain' and 'train' word, but it should be ok also for
    'val', 'test'.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.
    '''
 
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    df_fakes = [] 
    for phase in  ('train', 'valid', 'test'): 
            fake_data = ({'phase': [phase for _ in range(100)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(100)], 
                         'id': [int(index) for index in range(100)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(100)]}) 
            fake_data_dataframe = pd.DataFrame(fake_data)
            df_fakes.append(fake_data_dataframe) 

    df_fake_train, df_fake_val, df_fake_test = df_fakes

    # try if in the analysis name there is a "train" word
    csv_path_stem = ['constrain_train_dataset.csv',
                    'constrain_val_dataset.csv',
                    'constrain_test_dataset.csv']

    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)

    dfs_fake = read_three_dataframes(new_test_folder, csv_path_stem)

    assert(len(dfs_fake) == 3) # contains 3 different dfs

    # test if the dataset are correctly read
    df_train, df_valid, df_test = dfs_fake

    
    for dataframe, phase in zip([df_train, df_valid, df_test], ['train', 'valid', 'test']):
        # check that each dataframe has the same phase word(for df_train 'train', for df_valid 'valid', for df_test 'test')
        assert((dataframe['phase'] == dataframe['phase'][0]).all())
        # check that in df_train the phase is 'train', df_valid 'valid' and df_test 'test'
        assert((dataframe['phase'] == phase).all()) 

    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()

    new_test_folder.rmdir()


@given(valid_test_phase = st.sampled_from(('val', 'test', 'random_name_phase')))
def test_read_twodf(valid_test_phase):
    '''
    This function tests the correct reading of the data, if the data 
    are already split in two dataframes (train dataset and valid/test dataset).
    Regarding the train dataset: in the filename it must be a 'train' word.
    Instead, regarding the valid/test dataset no requests are needed.
    The name of this dataset, actually, could be anything.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.
    '''
 
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    df_fakes = [] 
    # try with train dataset (which is REQUESTED)
    # and VALID/TEST dataset
    for phase in  ('train', valid_test_phase): 
            fake_data = ({'phase': [phase for _ in range(100)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(100)], 
                         'id': [int(index) for index in range(100)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(100)]}) 
            fake_data_dataframe = pd.DataFrame(fake_data)
            df_fakes.append(fake_data_dataframe) 

    df_fake_train, df_fake_val = df_fakes


    csv_path_stem = ['fake_train_dataset.csv',
                    'fake_' + valid_test_phase +'_dataset.csv',]

    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    
    dfs_fake = read_two_dataframes(new_test_folder, csv_path_stem)

    assert(len(dfs_fake) == 2) # contains 2 different dfs

    # test if the dataset are correctly read
    df_train, df_valid = dfs_fake

    
    for dataframe, phase in zip([df_train, df_valid], ['train', valid_test_phase]):
        # check that each dataframe has the same phase word(for df_train 'train', for df_valid 'valid', for df_test 'test')
        assert((dataframe['phase'] == dataframe['phase'][0]).all())
        # check that in df_train the phase is 'train', df_valid 'valid' and df_test 'test'
        assert((dataframe['phase'] == phase).all()) 
    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()

    new_test_folder.rmdir()


@given(number_of_rows = st.integers(min_value = 0, max_value = 100))
def test_read_singledf(number_of_rows):
    '''
    This function tests the correct reading of the data, if the data 
    are given in a single csv file.
    In order to do so we will create a "fake" folder and we will place 
    some fake text data inside this folder.
    '''
 
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    df_fakes = [] 
    for phase in  ('alldata',): 
            fake_data = ({'phase': [phase for _ in range(number_of_rows)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(number_of_rows)], 
                         'id': [int(index) for index in range(number_of_rows)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(number_of_rows)]}) 
            fake_data_dataframe = pd.DataFrame(fake_data)
            df_fakes.append(fake_data_dataframe) 

    df_complete = df_fakes[0]

    csv_path_stem = ['Dataset_With_Random_Name.csv']

    df_complete.to_csv(new_test_folder / csv_path_stem[0], index=False)
    
    dfs_fake = read_data(new_test_folder)

    assert(len(dfs_fake) == 1) # contains one single dataframe

    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()

    new_test_folder.rmdir()

@given(phase = st.sampled_from(('train', 'val', 'test')))
def test_multiple_occurencies(phase):
    '''
    This function tests the correct handling if in a filename
    there are multiple occurencies for the words of interest.
    In our case the words of interest are: 'train', 'valid', 'test'.
    The idea behind this is function is that the train dataset corresponds 
    to the filename with the maximum number of occurencies for the 'train' word
    and the same for the other words of interest.
    '''
    name_max_count = 'trywith' + phase + '_' + phase + '.csv'

    list_names = ['trywith' + phase + '_train.csv',
                 'trywith' + phase + '_val.csv',
                 'trywith' + phase +  '_test.csv']
    
    word_to_count = phase
    name_with_max_count = handle_multiple_occurencies(list_names, word_to_count)

    assert(name_with_max_count == name_max_count)

    # now try with some whitespaces in the filenames

    name_max_count = 'trywith ' + phase + ' ' + phase + '.csv'

    list_names = ['trywith ' + phase + ' train.csv',
                 'trywith ' + phase + ' val.csv',
                 'trywith ' + phase +  ' test.csv']

    word_to_count = phase
    name_with_max_count = handle_multiple_occurencies(list_names, word_to_count)

    assert(name_with_max_count == name_max_count)

@given(phase = st.sampled_from(('Train', 'Val', 'Test')))
def test_multiple_occurencies_capital_letters(phase):
    '''
    This function tests the correct handling if in a filename
    there are multiple occurencies for the words of interest.
    In our case the words of interest are: 'train', 'valid', 'test'.
    The idea behind this is function is that the train dataset corresponds 
    to the filename with the maximum number of occurencies for the 'train' word
    and the same for the other words of interest.

    This test function tests in particular the behaviour of the hadle_multiple_occurencies 
    with the capital letters.
    '''
    name_max_count = 'trywith' + phase.lower() + '_' + phase + '.csv'

    list_names = ['trywith' + phase.lower() + '_Train.csv',
                 'trywith' + phase.lower() + '_Val.csv',
                 'trywith' + phase.lower() +  '_Test.csv']
    
    
    word_to_count = phase.lower()
    name_with_max_count = handle_multiple_occurencies(list_names, word_to_count)

    assert(name_with_max_count == name_max_count)


def test_read_4df():
    '''
    Test function to prove the correct working of the read_data function
    if inside the input folder there are more than three different files.

    If inside the input folder there are more than three files
    a ValueError is raised.
    '''

    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)

    df_fakes = [] 
    for phase in  ('train', 'val', 'test', 'newone'): 
            fake_data = ({'phase': [phase for _ in range(100)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(100)], 
                         'id': [int(index) for index in range(100)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(100)]}) 
            fake_data_dataframe = pd.DataFrame(fake_data)
            df_fakes.append(fake_data_dataframe) 

    df_fake_train, df_fake_val, df_fake_test, df_newone = df_fakes

    csv_path_stem = ['fake_train_dataset.csv',
                    'fake_val_dataset.csv',
                    'fake_test_dataset.csv',
                    'fakeboh_test_dataset.csv']
    df_fake_train.to_csv(new_test_folder / csv_path_stem[0], index=False)
    df_fake_val.to_csv(new_test_folder / csv_path_stem[1], index=False)
    df_fake_test.to_csv(new_test_folder / csv_path_stem[2], index=False)
    df_newone.to_csv(new_test_folder / csv_path_stem[3], index=False)

    with unittest.TestCase.assertRaises(unittest.TestCase, 
                                        expected_exception = ValueError):
        dfs_fake = read_data(new_test_folder)

    for csv_path in csv_path_stem:
        (new_test_folder / csv_path).unlink()    
    new_test_folder.rmdir()

