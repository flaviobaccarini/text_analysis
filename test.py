'''
All the tests function will be written in this module
'''
from explore import word_count_twitter, char_count_twitter
from read_write_data import read_data, write_data
#import numpy as np
import pandas as pd
from pathlib import Path

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
    pass
    new_test_folder = Path('test_folder')
    new_test_folder.mkdir(parents = True, exist_ok = True)
    #df_fake_train  


