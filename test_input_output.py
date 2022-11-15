'''
All the tests function will be written in this module
'''

from binary_classifier.read_write_data import read_data, write_data, read_three_dataframes
from binary_classifier.read_write_data import read_two_dataframes, handle_multiple_occurencies
from binary_classifier.read_write_data import split_dataframe, split_single_dataframe, split_two_dataframes

#import numpy as np
import pandas as pd
from pathlib import Path
import string    
import random # define the random module  


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
    df_fake_train.to_csv(new_test_folder / 'fake_train_dataset.csv', index=False)
    df_fake_val.to_csv(new_test_folder / 'fake_val_dataset.csv', index=False)
    df_fake_test.to_csv(new_test_folder / 'fake_test_dataset.csv', index=False)

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
    (new_test_folder / 'fake_train_dataset.csv').unlink()
    (new_test_folder / 'fake_val_dataset.csv').unlink()
    (new_test_folder / 'fake_test_dataset.csv').unlink()

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

    write_data(df_fakes_tuple, name_folder)

    csv_paths = list(path_output.glob('**/*.csv'))
    
    assert(len(csv_paths) == 3) # writing function writes the three different csv files containing the datasets

    for csv_path in csv_paths:
        csv_path.unlink()
    

    write_data(df_fakes_list, name_folder) # it works with both tuple or list

    csv_paths = list(path_output.glob('**/*.csv'))
    
    assert(len(csv_paths) == 3) # writing function writes the three different csv files containing the datasets

    for csv_path in csv_paths:
        csv_path.unlink()
    
    path_output.rmdir()