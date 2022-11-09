'''
All the tests function will be written in this module
'''
from explore import word_count_twitter, char_count_twitter
from read_write_data import read_data, write_data
#import numpy as np
import pandas as pd
from pathlib import Path
import string    
import random # define the random module  


def test_word_count():
    '''
    This function is used to test how the counting of the words is done.
    '''

    test_string = ['Hello, this is a test string']
    number_of_words = word_count_twitter(test_string)

    assert(len(number_of_words) == 1) # contain only one single sentence
    assert(number_of_words[0] == 6) # the first sentence contains 6 word

    test_string.append('Hello world, this is the number 2 #test string.')
    number_of_words = word_count_twitter(test_string)

    assert(len(number_of_words) == 2) # now the list contains two different sentences
    assert(number_of_words[1] == 9) # the second sentence is composed by 7 words

def test_word_count_input_type():
    '''
    This function is used to test if 
    different input type for the word count function can work,
    '''

    list_test_strings = ['Test string', 'Test pandas series']
    series_test_string = pd.Series(list_test_strings)
    tuple_test_string = tuple(list_test_strings)

    number_of_words = word_count_twitter(list_test_strings)

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word

    number_of_words = word_count_twitter(series_test_string)

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word

    number_of_words = word_count_twitter(tuple_test_string)

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word


def test_char_count():
    '''
    This function is used to test how the counting of the characters is done.
    '''

    test_string = ['Hello, this is a test string']
    number_of_chars = char_count_twitter(test_string)

    assert(len(number_of_chars) == 1) # contain only one single sentence
    assert(number_of_chars[0] == 28) # the first sentence contains 28 characters

    test_string.append('Hello world, this is another test string.')
    number_of_chars = char_count_twitter(test_string)

    assert(len(number_of_chars) == 2) # now the list contains two different sentences
    assert(number_of_chars[1] == 41) # the second sentence is composed by 41 characters

def test_char_count_input_type():
    '''
    This function is used to test if 
    different input type for the characters count function can work,
    '''

    list_test_strings = ['1st test string', '#Test pandas series', '#Wikipedia site: https://wikipedia.org']
    series_test_string = pd.Series(list_test_strings)
    tuple_test_string = tuple(list_test_strings)

    number_of_chars = char_count_twitter(list_test_strings)

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[0] == 15) # the first sentence contains 15 chars

    number_of_chars = char_count_twitter(series_test_string)

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[1] == 19) # the second sentence contains 19 chars

    number_of_chars = char_count_twitter(tuple_test_string) # tuple

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[2] == 38) # the third sentence contains 38 chars


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