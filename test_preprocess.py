import numpy as np
import pandas as pd
from pathlib import Path
import string    
import random # define the random module  
from hypothesis import strategies as st
from hypothesis import given
from preprocess import clean_dataframe, lower_strip
from preprocess import remove_urls_tags, remove_emojis, remove_quotmarks_underscore




@given(column_names = st.lists(st.text(min_size = 1, max_size = 10), min_size = 2, max_size = 2),
all_sentences = st.lists(st.lists(st.text(min_size=0, max_size=20), min_size = 0, max_size = 100),
                         min_size = 3, max_size = 3))
def test_clean_df_colname(column_names, all_sentences):
    #TODO: QUESTA FUNZIONE SI PUÒ SNELLIRE VOLENDO TOGLIENDO LE all_sentences COME ARGOMENTO...
    '''
    Test function to prove the correctly rename of the columns in the new dataset.
    The initial dataset contained at least two columns with the name given in column_names.
    These two columns contained: 1) text 2) labels.
    The final dataset contain only two columns with name "text" and "label" that correspond
    to the text and the label.
    '''

    df_fakes = []
    unique_labels = ['real', 'fake']

    for sentence in (all_sentences):
        labels = [unique_labels[random.randrange(len(unique_labels))] for _ in range(len(sentence))]
        fake_data =({column_names[0]: sentence, column_names[1]: labels})
        fake_data_dataframe = pd.DataFrame(fake_data)
        df_fakes.append(fake_data_dataframe)

    dfs_cleaned = clean_dataframe(df_fakes, column_names)

    assert(len(dfs_cleaned) == 3) # check that there are three dataframes
    for df_clean in dfs_cleaned:
        assert(df_clean.columns.values.tolist() == ['text', 'label']) # check name of colums are correct
        if len(df_clean) != 0: # check if in the label column there are actually the labels
            assert(list(df_clean['label']).count('real') > 0 or list(df_clean['label']).count('fake') > 0 )



@given(column_names = st.lists(st.text(min_size = 1, max_size = 10), min_size = 2, max_size = 2),
all_sentences = st.lists(st.lists(st.text(min_size=0, max_size=20), min_size = 0, max_size = 100),
                         min_size = 3, max_size = 3))
def test_clean_empty_df(column_names, all_sentences):
    #TODO: QUESTA FUNZIONE SI PUÒ SNELLIRE VOLENDO TOGLIENDO LE column_names COME ARGOMENTO...
    '''
    Test function to prove the correctly cleaning of the dataframe.
    The initial dataset can be composed of empty cells, or cells with no text inside (only '').
    This kind of data are not so meaningful, so the idea behind this function is to remove these rows.
    The final dataset contain only rows with text.
    '''
    df_fakes = []
    unique_labels = ['real', 'fake',]

    for sentence in (all_sentences):
        labels = [unique_labels[random.randrange(len(unique_labels))] for _ in range(len(sentence))]
        fake_data =({column_names[0]: sentence, column_names[1]: labels})
        fake_data_dataframe = pd.DataFrame(fake_data)
        df_fakes.append(fake_data_dataframe)

    dfs_cleaned = clean_dataframe(df_fakes, column_names)

    assert(len(dfs_cleaned) == 3) # check that there are three dataframes
    for df_clean in dfs_cleaned:
        assert(len(np.where(pd.isnull(df_clean))[0]) == 0) # no empty cells for text
        assert(len(np.where(pd.isnull(df_clean))[1]) == 0) # no empty cells for label
        assert( len(np.where(df_clean.applymap(lambda x: x == ''))[0]) == 0) # no cells '' for text
        assert( len(np.where(df_clean.applymap(lambda x: x == ''))[1]) == 0) # no cells '' for label

        if len(df_clean )> 0:
            text_count = map(len, df_clean['text'])
            assert(min(text_count) >= 1)

def test_lower():
    '''
    This test function tests the correct working of the lower_strip function.
    In particular, in this test function only the lower part is tested.
    '''
    capital_text = 'HELLO WORLD'
    text_lower = lower_strip(capital_text)
    another_text = 'Hello World'
    another_text_lower = lower_strip(another_text)
    text_already_low = 'hello world'
    lower_text_already_low = lower_strip(text_already_low)

    assert(text_lower == 'hello world')
    assert(another_text_lower == 'hello world')
    assert(lower_text_already_low == 'hello world')

def test_strip():
    '''
    This test function tests the correct working of the lower_strip function.
    In particular, in this test function the stripping part is tested.
    '''
    text_to_strip = '    hello world'
    text_stripped = lower_strip(text_to_strip)

    text_to_strip2 = 'hello world    '
    text_stripped2 = lower_strip(text_to_strip2)

    text_to_strip3 = '     Hello         World     ' 
    text_stripped3 = lower_strip(text_to_strip3) # strip the string and eliminate white spaces

    assert(text_stripped == 'hello world')
    assert(text_stripped3 == 'hello world')
    assert(text_stripped2 == 'hello world')    

@given(text = st.text(min_size = 0, max_size = 10))
def test_lower_strip(text):
    '''
    This test function tests the correct working of the lower_strip function
    with some random text given as input.
    '''
    # 1) lower the text; 2) strip the text; 3) remove whitespaces
    correct_text = " ".join(text.lower().split())
    text_from_function = lower_strip(text)
    assert(text_from_function == correct_text)

def test_remove_url():
    '''
    This test function tests the correct working of the remove_url_tag function.
    In particular, in this test function only the part about urls is tested.
    '''
    text_with_url = 'This is wikipedia site: https://en.wikipedia.org/wiki'
    text_without_url = remove_urls_tags(text_with_url)

    only_url = 'https://en.wikipedia.org/wiki'
    no_url = remove_urls_tags(only_url)

    url_with_space = '   https://en.wikipedia.org/wiki   '
    six_whitespaces = remove_urls_tags(url_with_space)

    text_no_urls = 'Hello World'
    text_no_urls_func = remove_urls_tags(text_no_urls)
    assert(text_without_url == 'This is wikipedia site: ')
    assert(no_url == '')
    assert(six_whitespaces == '      ')
    assert(text_no_urls_func == text_no_urls)


def test_remove_tag():
    '''
    This test function tests the correct working of the remove_url_tag function.
    In particular, in this test function only the part about tags is tested.
    '''
    text_with_tag = '<TEXT> Hello World <TEXT>'
    text_without_tag = remove_urls_tags(text_with_tag)

    only_tag = '<HELLO WORLD>'
    no_tag = remove_urls_tags(only_tag)

    text_no_tag = 'Hello World'
    text_no_tag_func = remove_urls_tags(text_no_tag)
    assert(text_without_tag == ' Hello World ')
    assert(no_tag == '')
    assert(text_no_tag_func == text_no_tag)


def test_remove_emojis():
    '''
    This test function tests the correct working of the remove_emojis function.
    '''
    text = 'Hello world \U0001f600'
    text_no_emoji = remove_emojis(text)

    text_example2 = 'This is a dog: \U0001F436'
    text_no_emoji2 = remove_emojis(text_example2)

    text_example3 = 'This is a rainbow: \U0001F308'
    text_no_emoji3 = remove_emojis(text_example3)

    text_example4 = '\U0000274C this is a cross mark'
    text_no_emoji4 = remove_emojis(text_example4)

    text_example5 = 'This is the number three: \U00000033'
    text_no_emoji5 = remove_emojis(text_example5)

    text_example6 = 'This is the letter B: \U0001F171'
    text_no_emoji6= remove_emojis(text_example6)

    text_example7 = 'This is SOS: \U0001F198'
    text_no_emoji7 = remove_emojis(text_example7)

    text_example8 = 'This is red circle: \U0001F534'
    text_no_emoji8 = remove_emojis(text_example8)

    assert(text_no_emoji == 'Hello world ')
    assert(text_no_emoji2 == 'This is a dog: ')
    assert(text_no_emoji3 == 'This is a rainbow: ')
    assert(text_no_emoji4 == ' this is a cross mark')
    assert(text_no_emoji5 == 'This is the number three: ')
    assert(text_no_emoji6 == 'This is the letter B: ')
    assert(text_no_emoji7 == 'This is SOS: ')
    assert(text_no_emoji8 == 'This is red circle: ')

def test_remove_quotation_marks():
    '''
    This test function tests the correct working of the remove_quotmarks_underscore function.
    In particular, this test functions tests that the quotation marks or the apostrophes are eliminated from the sentences
    with leaving one space between words. In this way, if we have something like "I've done something...", 
    it remains just "I ve done something..." and with the lemmatizer and the stopwords it's possible to remove both
    "I" and "ve".
    '''
    text_example1 = "Hello world, it's me"
    text_no_quot_mark1 = remove_quotmarks_underscore(text_example1)

    text_example2 = '"Hello world" he said'
    text_no_quot_mark2 = remove_quotmarks_underscore(text_example2)

    assert(text_no_quot_mark1 == 'Hello world, it s me')
    assert(text_no_quot_mark2 == ' Hello world  he said')


def test_rm_underscore():
    '''
    This test function tests the correct working of the remove_quotmarks_underscore function.
    In particular, this test functions tests that underscores are eliminated from the sentences
    with leaving one spaces between words.
    '''
    text_example1 = "__Hello world__"
    text_no_underscore1 = remove_quotmarks_underscore(text_example1)

    text_example2 = 'Hello_world'
    text_no_underscore2 = remove_quotmarks_underscore(text_example2)

    assert(text_no_underscore1 == ' Hello world ')
    assert(text_no_underscore2 == 'Hello world')

