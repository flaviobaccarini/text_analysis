'''
In this file there are test functions to test the 
functions inside the preanalysis module.
'''
from text_analysis.preanalysis import average_word_or_chars
from text_analysis.preanalysis import word_count_text, char_count_text
import pandas as pd
import numpy as np 
import unittest

def test_word_count_number_of_sentences():
    '''
    Test function to test word_count_text behaviour.
    In this test function it's tested how many sentences
    there are inside the input list.

    Given:
    ======
    test_strings: list[str]
                  Sequence of text for testing.

    Tests:
    =======
            How many sentences there are inside the 
            initial list (test_strings).                
    '''

    test_strings = ['Hello, this is a test string', 'My name is FLavio'] # two sentences
    number_of_words = word_count_text(test_strings)
    assert(len(number_of_words) == 2) # two sentences

def test_word_count():
    '''
    Test function to test word_count_text behaviour.
    In this test function it's tested how many words
    there are inside the input string.

    Given:
    ======
    test_string: list[str]
                 Input text string.

    Tests:
    =======
            How many words there are inside the 
            initial string (test_string).                
    '''

    test_string = ['Hello world'] # two words sentence
    number_of_words = word_count_text(test_string)
    assert(number_of_words[0] == 2) # 2 words

def test_word_count_tuple_input():
    '''
    Test function to test word_count_text behaviour.
    This function is used to test if a tuple 
    for the word_count_text function can work.
        
    Given:
    ======
    test_string: tuple[str]
                 Tuple of text for testing.

    Tests:
    =======
            If the word_count_text can work with
            tuple as input.           
    '''
    test_string = ('',)
    number_of_words = word_count_text(test_string)
    assert(number_of_words[0] == 0)

def test_word_count_series_input():
    '''
    Test function to test word_count_text behaviour.
    This function is used to test if a pandas Series 
    for the word_count_text function can work.
        
    Given:
    ======
    test_string: pd.Series[str]
                 Pandas Series of text for testing.

    Tests:
    =======
            If the word_count_text can work with
            pd.Series as input.           
    '''

    test_string = pd.Series([''])
    number_of_words = word_count_text(test_string)
    assert(number_of_words[0] == 0)
    
def test_word_count_nparray_input():
    '''
    Test function to test word_count_text behaviour.
    This function is used to test if a numpy array 
    for the word_count_text function can work.
        
    Given:
    ======
    test_string: np.array[str]
                 Numpy Array of text for testing.

    Tests:
    =======
            If the word_count_text can work with
            np.array as input.           
    '''

    test_string = np.array([''])
    number_of_words = word_count_text(test_string)
    assert(number_of_words[0] == 0)

def test_char_count_number_of_sentences():
    '''
    Test function to test char_count_text.
    In this test function it's tested how many sentences
    there are inside the input list.

    Given:
    ======
    test_strings: list[str]
                  Sequence of text for testing.

    Tests:
    =======
            How many sentences there are inside the 
            initial list (test_strings).                
    '''
    test_string = ['Hello, test string',
                   'Hello world',
                   'Another string'] # three sentences
    number_of_chars = char_count_text(test_string)
    assert(len(number_of_chars) == 3) # three sentences
 
def test_char_count():
    '''
    Test function to test char_count_text.
    In this test function it's tested how many characters
    there are inside the input string.

    Given:
    ======
    test_string: list[str]
                 Input text string.

    Tests:
    =======
            How many characters there are inside the 
            initial input string (test_string).                
    '''
    test_string = ['Hello!'] # Hello! is composed by 6 chars
    number_of_chars = char_count_text(test_string)
    assert(number_of_chars[0] == 6) 

def test_char_count_tuple_input():
    '''
    Test function to test char_count_text.
    In this test function it's tested if a tuple of string
    can work as input to char_count_text function.
    
    Given:
    ======
    test_string: tuple[str]
                 Tuple of text for testing.
    
    Tests:
    =======
            if the char_count_text can work with tuple
            of string as input.
    '''
    test_string = ('',)
    number_of_chars = char_count_text(test_string)
    assert(number_of_chars[0] == 0)

def test_char_count_series_input():
    '''
    Test function to test char_count_text.
    In this test function it's tested if a pandas Series
    of string can work as input to char_count_text function.
    
    Given:
    ======
    test_string: pd.Series[str]
                 Pandas Series of text for testing.
    
    Tests:
    =======
            if the char_count_text can work with pandas
            Series of string as input.
    '''
    test_string = pd.Series([''])
    number_of_chars = char_count_text(test_string)
    assert(number_of_chars[0] == 0)

def test_char_count_nparray_input():
    '''
    Test function to test char_count_text.
    In this test function it's tested if a numpy array
    of string can work as input to char_count_text function.
    
    Given:
    ======
    test_string: np.array[str]
                 Pandas Series of text for testing.
    
    Tests:
    =======
            if the char_count_text can work with numpy
            array of string as input.
    '''
    test_string = np.array([''])
    number_of_chars = char_count_text(test_string)
    assert(number_of_chars[0] == 0)

def test_char_count_empty_list():
    '''
    This function tests the correct working of the counting function 
    for characters (char_count_text), when an empty list
    is passed as input to the function.

    Given:
    ======
    empty_list_string: list
                       Empty list that will be passed to the function.
    
    Tests:
    ======
            We expect that a ValueError is raised by the function,
            because the list passed as input is empty.
    '''
    empty_list_string = []
    with unittest.TestCase.assertRaises(unittest.TestCase,
                                            expected_exception = ValueError):
            chars_count = char_count_text(empty_list_string)

def test_word_count_empty_list():
    '''
    This function tests the correct working of the counting function 
    for words (word_count_text), when an empty list
    is passed as input to the function.

    Given:
    ======
    empty_list_string: list
                       Empty list that will be passed to the function.
    
    Tests:
    ======
            We expect that a ValueError is raised by the function,
            because the list passed as input is empty.
    '''
    empty_list_string = []
    with unittest.TestCase.assertRaises(unittest.TestCase,
                                            expected_exception = ValueError):
            words_count = word_count_text(empty_list_string)

def test_avg_word_single_label():
    '''
    This function test the correct working of the 
    average function for a single string label (average_word_or_chars).
    average_word_or_chars takes as input the labels and the counts
    and the output is a dict with keys mapped to the average counts
    for each single label. 

    Given:
    =======
    labels: list[str]
            The sequence of the labels. In this test function
            only one single label.
    
    counts: list[int]
            Sequence of counts. In our case, it represents
            the count number of words or characters.
    
    Tests:
    ======
            Test if the average for the counts is correct 
            for one single label.
    '''
    labels = ['A', 'A', 'A']
    counts = [10, 15, 20]
    avg = average_word_or_chars(labels, counts)
    assert(avg['A'] == 15)

def test_avg_word_twolabels():
    '''
    This function test the correct working of the 
    average function for two string labels (average_word_or_chars).
    average_word_or_chars takes as input the labels and the counts
    and the output is a dict with keys mapped to the average counts
    for each single label. 

    Given:
    =======
    labels: list[str]
            The sequence of the labels. In this test function
            only two labels.
    
    counts: list[int]
            Sequence of counts. In our case, it represents
            the count number of words or characters.
    
    Tests:
    ======
            Test if the average for the counts is correct for
            each label.
    '''
    labels = ['A', 'B', 'B', 'A']
    counts = [10, 40, 20, 20]
    avg = average_word_or_chars(labels, counts)
    assert(avg['A'] == 15)
    assert(avg['B'] == 30)

def test_avg_word_threelabels():
    '''
    This function test the correct working of the 
    average function for three string labels (average_word_or_chars).
    average_word_or_chars takes as input the labels and the counts
    and the output is a dict with keys mapped to the average counts
    for each single label. 

    Given:
    =======
    labels: list[str]
            The sequence of the labels. In this test function
            only three labels.
    
    counts: list[int]
            Sequence of counts. In our case, it represents
            the count number of words or characters.
    
    Tests:
    ======
            Test if the average for the counts is correct for
            each label.
    '''
    labels = ['A', 'B', 'C', 'A']
    counts = [10, 40, 50, 20]
    avg = average_word_or_chars(labels, counts)
    assert(avg['A'] == 15)
    assert(avg['B'] == 40)
    assert(avg['C'] == 50)

def test_avg_floats():
    '''
    This function test the correct working of the 
    average function for a single label (average_word_or_chars).
    average_word_or_chars takes as input the labels and the counts
    and the output is a dict with keys mapped to the average counts
    for each single label. 

    In this particular test function it's tested if the function
    can properly work also with float numbers.

    Given:
    =======
    labels: list[str]
            The sequence of the labels. In this test function
            only one single label.
    
    counts: list[float]
            Sequence of float numbers. 
    
    Tests:
    ======
            Test if the average for the counts is correct,
            even though the counts are float numbers.
    '''
    labels = ['A', 'A']
    counts = [2.25, 2.75]
    avg = average_word_or_chars(labels, counts)
    assert(avg['A'] == 2.5)
