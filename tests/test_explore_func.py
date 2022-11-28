'''
This test script is written in order to test the exploring function in the explore module.
'''
from binary_classifier.preanalysis import average_word_or_chars, word_count_text, char_count_text
from hypothesis import strategies as st
from hypothesis import given
import pandas as pd
import random
import numpy as np 
import unittest
def test_word_count():
    '''
    This function is used to test how the counting of the words is done.
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
    This function is used to test if 
    different input type for the word count function can work,
    '''

    list_test_strings = ['Test string', 'Test pandas series']
    series_test_string = pd.Series(list_test_strings)
    tuple_test_string = tuple(list_test_strings)

    number_of_words = word_count_text(list_test_strings)

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word

    number_of_words = word_count_text(series_test_string)

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word

    number_of_words = word_count_text(tuple_test_string)

    assert(len(number_of_words) == 2) # contain only one two sentences
    assert(number_of_words[1] == 3) # the second sentence contains 3 word


def test_char_count():
    '''
    This function is used to test how the counting of the characters is done.
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
    This function is used to test if 
    different input type for the characters count function can work,
    '''

    list_test_strings = ['1st test string', '#Test pandas series', '#Wikipedia site: https://wikipedia.org']
    series_test_string = pd.Series(list_test_strings)
    tuple_test_string = tuple(list_test_strings)

    number_of_chars = char_count_text(list_test_strings)

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[0] == 15) # the first sentence contains 15 chars

    number_of_chars = char_count_text(series_test_string)

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[1] == 19) # the second sentence contains 19 chars

    number_of_chars = char_count_text(tuple_test_string) # tuple

    assert(len(number_of_chars) == 3) # contain 3 sentences
    assert(number_of_chars[2] == 38) # the third sentence contains 38 chars


@given(text = st.lists(st.text(max_size = 20)))
def test_char_count(text):
    '''
    This function tests the correct working of the counting function for characaters.
    '''
    if len(text) == 0:
        with unittest.TestCase.assertRaises(unittest.TestCase, expected_exception = ValueError):
            chars_count = char_count_text(text)
    else:
        chars_count = char_count_text(text)
        for sentence, char_count in zip(text, chars_count):
            assert(len(sentence) == char_count)

    
@given(num_of_sentences = st.integers(min_value=0, max_value=10),
       list_word = st.lists(st.text(max_size = 20)))
def test_word_count(num_of_sentences, list_word):
    '''
    This function tests the correct working of the counting function for words.
    ''' 
    all_texts = []
    for _ in range(num_of_sentences):
        text = ' '.join(random.choices(list_word, k = len(list_word)))
        all_texts.append(text)

    if len(all_texts) == 0:
        with unittest.TestCase.assertRaises(unittest.TestCase, expected_exception = ValueError):
            words_count = word_count_text(all_texts)
    else:        
        words_count = word_count_text(all_texts)
        for word_count, text in zip(words_count, all_texts):
            assert(len(text.split()) == word_count)

test_word_count()
def test_average_word_func_one_two_lab():
    '''
    This function test the correct working of the average function for one or two string labels.
    '''
    unique_labels = ['real']
    labels = [unique_labels[random.randrange(len(unique_labels))] for _ in range(100)]
    counts = [random.randrange(0, 100) for _ in range(100)]

    avg_from_funct = average_word_or_chars(labels, counts)

    assert(avg_from_funct['real'] == np.mean(counts)) # one single label
    

    unique_labels = ['real', 'fake'] # two labels
    labels = [unique_labels[random.randrange(len(unique_labels))] for _ in range(100)]
    labels_sorted = sorted(labels)

    real_count = labels.count('real')
    fake_count = labels.count('fake')
    assert(real_count + fake_count == 100)

    counts_real = [random.randrange(0, 100) for _ in range(real_count)]
    counts_fake = [random.randrange(0, 100) for _ in range(fake_count)]
    all_counts = counts_fake + counts_real
    assert(len(all_counts) == 100)

    mean_real = np.mean(counts_real)
    mean_fake = np.mean(counts_fake)

    df = pd.DataFrame({'label': labels_sorted, 'counts': all_counts})
    df = df.sample(frac=1).reset_index(drop=True)
    labels = list(df['label'])
    all_counts = df['counts']


    avg_from_funct = average_word_or_chars(labels, all_counts)

    assert(avg_from_funct['real'] == mean_real)
    assert(avg_from_funct['fake'] == mean_fake)

@given(number_of_tot_labels = st.integers(min_value = 100, max_value = 150),
       labels = st.lists(st.integers(), min_size=3, max_size=20))   
def test_average_word_func_multiple_lab(number_of_tot_labels, labels):
    '''
    This function test the correct working of the average function for multiple integer labels.
    '''
    unique_labels = np.unique(labels)
    labels = [unique_labels[random.randrange(len(unique_labels))] for _ in range(number_of_tot_labels)]
    labels_sorted = sorted(labels)
    count_label_list = []
    for label in unique_labels:
        count_label = labels.count(label)
        count_label_list.append(count_label)
    
    assert(sum(count_label_list) == number_of_tot_labels)
    counts_generated = []
    for number_of_count_per_label in count_label_list:
        count_generated = [random.randrange(0, 100) for _ in range(number_of_count_per_label)]
        counts_generated.append(count_generated)

    all_counts = [item for sublist in counts_generated for item in sublist]
    assert(len(all_counts) == number_of_tot_labels)

    averages = []
    for count_generated in counts_generated:
        averages.append(np.mean(count_generated))

    df = pd.DataFrame({'label': labels_sorted, 'counts': all_counts})
    df = df.sample(frac=1).reset_index(drop=True)
    labels = list(df['label'])
    all_counts = (df['counts'])

    avg_from_funct = average_word_or_chars(labels, all_counts)
    
    for label, average in zip(unique_labels, averages):
        assert(avg_from_funct[label] == average)
 