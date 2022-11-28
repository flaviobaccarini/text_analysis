'''
In this file there are different test functions to test the 
preanalysis module.
'''
from binary_classifier.preanalysis import average_word_or_chars
from binary_classifier.preanalysis import word_count_text, char_count_text
from hypothesis import strategies as st
from hypothesis import given
import pandas as pd
import random
import numpy as np 
import unittest

def test_word_count():
    '''
    Test function to test word_count_text.
    We can see how the counting of the words is done.
    '''

    test_string = ['Hello, this is a test string']
    number_of_words = word_count_text(test_string)

    assert(len(number_of_words) == 1) # contain only one single sentence
    assert(number_of_words[0] == 6) # the first sentence contains 6 word

    test_string.append('Hello world, this is the number 2 #test string.')
    number_of_words = word_count_text(test_string)

    assert(len(number_of_words) == 2) # now the list contains two different sentences
    assert(number_of_words[1] == 9) # the second sentence is composed by 7 words

def test_word_count_input_type():
    '''
    This function is used to test if ifferent input type
    for the word count function can work (word_count_text).
    '''

    list_test_strings = ['Test string', 'Test pandas series']
    series_test_string = pd.Series(list_test_strings)
    tuple_test_string = tuple(list_test_strings)
    numpy_test_string = np.array(list_test_strings)

    number_of_words = word_count_text(list_test_strings) # list

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word

    number_of_words = word_count_text(series_test_string) # pd series

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word

    number_of_words = word_count_text(tuple_test_string) # tuple

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word

    number_of_words = word_count_text(numpy_test_string) # np array

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word

def test_char_count():
    '''
    Test function to test char_count_text.
    We can see how the counting of the characters is done.
    '''
    test_string = ['Hello, this is a test string']
    number_of_chars = char_count_text(test_string)

    assert(len(number_of_chars) == 1) # contain only one single sentence
    assert(number_of_chars[0] == 28) # the first sentence contains 28 characters

    test_string.append('Hello world, this is another test string.')
    number_of_chars = char_count_text(test_string)

    assert(len(number_of_chars) == 2) # now the list contains two different sentences
    assert(number_of_chars[1] == 41) # the second sentence is composed by 41 characters

def test_char_count_input_type():
    '''
    This function is used to test if different input type
    for the characters count function can work (char_count_text).
    '''

    list_test_strings = ['1st test string', '#Test pandas series', 
                         '#Wikipedia site: https://wikipedia.org']
    series_test_string = pd.Series(list_test_strings)
    tuple_test_string = tuple(list_test_strings)
    numpy_test_string = np.array(list_test_strings)

    number_of_chars = char_count_text(list_test_strings) # list

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[0] == 15) # the first sentence contains 15 chars

    number_of_chars = char_count_text(series_test_string) # pd series

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[1] == 19) # the second sentence contains 19 chars

    number_of_chars = char_count_text(tuple_test_string) # tuple

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[2] == 38) # the third sentence contains 38 chars

    number_of_chars = char_count_text(numpy_test_string) # np array

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[2] == 38) # the third sentence contains 38 chars



@given(text = st.lists(st.text(max_size = 20)))
def test_char_count(text):
    '''
    This function tests the correct working of the counting function 
    for random characaters (char_count_text).

    @given:
    ========
    text: list[str]
          Random text to use as input to char_count_text.
    '''
    # test the function
    if len(text) == 0:
        with unittest.TestCase.assertRaises(unittest.TestCase,
                                            expected_exception = ValueError):
            chars_count = char_count_text(text)
    else:
        chars_count = char_count_text(text)
        for sentence, char_count in zip(text, chars_count):
            assert(len(sentence) == char_count)

    
@given(num_of_sentences = st.integers(min_value=0, max_value=10),
       list_word = st.lists(st.text(max_size = 20)))
def test_word_count(num_of_sentences, list_word):
    '''
    This function tests the correct working of the counting function
    word_count_text for random words.

    @given:
    ========
    num_of_sentences: int
                      Total number of sentences.
    
    list_word: list[str]
               The vocabulary to create some random sentences.
    ''' 
    all_texts = []
    # generate some random text
    for _ in range(num_of_sentences):
        text = ' '.join(random.choices(list_word, k = len(list_word)))
        all_texts.append(text)

    # test the function
    if len(all_texts) == 0:
        with unittest.TestCase.assertRaises(unittest.TestCase, 
                                            expected_exception = ValueError):
            words_count = word_count_text(all_texts)
    else:        
        words_count = word_count_text(all_texts)
        for word_count, text in zip(words_count, all_texts):
            assert(len(text.split()) == word_count)

def test_average_word_func_two_labels():
    '''
    This function test the correct working of the 
    average function for two string labels (average_word_or_chars).
    '''
    # two labels
    unique_labels = ['real', 'fake']

    # generate some random labels
    labels = [unique_labels[random.randrange(len(unique_labels))] for _ in range(100)]
    # sorting labels makes the average computation easier
    labels_sorted = sorted(labels)

    # count how many occurencies for each label
    real_count = labels.count('real')
    fake_count = labels.count('fake')

    # we expect that the total occurencies for labels is equal to 100
    assert(real_count + fake_count == 100)

    # generate some random counts for the occurencies we computed before for each label
    counts_real = [random.randrange(0, 100) for _ in range(real_count)]
    counts_fake = [random.randrange(0, 100) for _ in range(fake_count)]
    all_counts = counts_fake + counts_real
    assert(len(all_counts) == 100)

    # compute the mean for each label
    mean_real = np.mean(counts_real)
    mean_fake = np.mean(counts_fake)

    # shuffle the data with pandas
    df = pd.DataFrame({'label': labels_sorted, 'counts': all_counts})
    df = df.sample(frac=1).reset_index(drop=True)
    labels = list(df['label'])
    all_counts = df['counts']

    # test the function
    avg_from_funct = average_word_or_chars(labels, all_counts)

    assert(avg_from_funct['real'] == mean_real)
    assert(avg_from_funct['fake'] == mean_fake)

@given(labels_list = st.lists(st.integers(), min_size=3, max_size=20))   
def test_average_word_func_multiple_lab(labels_list):
    '''
    This function test the correct working of
    the average function for multiple integer labels (average_word_or_chars).

    @given:
    ========
    labels_list: list[int]
                 List of all the possible labels.
    '''
    unique_labels = np.unique(labels_list)
    number_of_tot_labels = 500
    # generate randomly some labels
    labels = [unique_labels[random.randrange(len(unique_labels))]
                                         for _ in range(number_of_tot_labels)]
    # sorting labels makes average computation easier
    labels_sorted = sorted(labels)
    count_label_list = []
    # compute how many occurencies for each label
    for label in unique_labels:
        count_label = labels.count(label)
        count_label_list.append(count_label)
    
    assert(sum(count_label_list) == number_of_tot_labels)
    counts_generated = []
    # generate randomly some counts for each label
    for number_of_count_per_label in count_label_list:
        count_generated = [random.randrange(0, 100) 
                                        for _ in range(number_of_count_per_label)]
        counts_generated.append(count_generated)

    all_counts = [item for sublist in counts_generated for item in sublist]
    # we expect number_of_tot_labels of counts
    assert(len(all_counts) == number_of_tot_labels)

    averages = []
    # compute the average for each label
    for count_generated in counts_generated:
        averages.append(np.mean(count_generated))

    # shuffle labels with pandas
    df = pd.DataFrame({'label': labels_sorted, 'counts': all_counts})
    df = df.sample(frac=1).reset_index(drop=True)
    labels = list(df['label'])
    all_counts = (df['counts'])

    # test the function
    avg_from_funct = average_word_or_chars(labels, all_counts)
    
    for label, average in zip(unique_labels, averages):
        assert(avg_from_funct[label] == average)

