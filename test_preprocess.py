import numpy as np
import pandas as pd
from pathlib import Path
import string    
import random # define the random module  
from hypothesis import strategies as st
from hypothesis import given
from preprocess import clean_dataframe, lower_strip
from preprocess import remove_urls_tags, remove_emojis, remove_quotmarks_underscore
from preprocess import remove_noalphanum, stopword, clean_text



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
    capital_texts = ['HELLO WORLD',
                     'Hello World',
                     'hello world']
    lower_texts = []
    for capital in capital_texts:
        text_lower = lower_strip(capital)
        lower_texts.append(text_lower)

    assert(lower_texts[0] == 'hello world')
    assert(lower_texts[1] == 'hello world')
    assert(lower_texts[2] == 'hello world')

def test_strip():
    '''
    This test function tests the correct working of the lower_strip function.
    In particular, in this test function the stripping part is tested.
    The lower_strip function eliminates also useless whitespaces. 
    '''
    texts_to_strip = ['    hello world',
                      'hello world    ',
                      '     Hello         World     ' ]
    texts_stripped = []
    for text_to_strip in texts_to_strip:
        text_stripped = lower_strip(text_to_strip)
        texts_stripped.append(text_stripped)

    assert(texts_stripped[0] == 'hello world')
    assert(texts_stripped[1] == 'hello world')
    assert(texts_stripped[2] == 'hello world')    


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

    texts_with_url = ['This is wikipedia site: https://en.wikipedia.org/wiki',
                      'https://en.wikipedia.org/wiki',
                      '   https://en.wikipedia.org/wiki   ',
                      'Hello World']
    text_nourls = []
    for text_with_url in texts_with_url:
        text_without_url = remove_urls_tags(text_with_url)
        text_nourls.append(text_without_url)
    
    assert(text_nourls[0] == 'This is wikipedia site: ')
    assert(text_nourls[1] == '')
    assert(text_nourls[2] == '      ')
    assert(text_nourls[3] == 'Hello World')

def test_remove_tag():
    '''
    This test function tests the correct working of the remove_url_tag function.
    In particular, in this test function only the part about tags is tested.
    '''
    texts_with_tag = ['<TEXT> Hello World <TEXT>',
                      '<HELLO WORLD>',
                      'Hello World',
                      '5 > 4 and 4 < 3',
                      '4 < 3 but 5 > 3']
    
    text_no_tags = []
    for text_with_tag in texts_with_tag:
        text_without_tag = remove_urls_tags(text_with_tag)
        text_no_tags.append(text_without_tag)

    assert(text_no_tags[0] == ' Hello World ')
    assert(text_no_tags[1] == '')
    assert(text_no_tags[2] == 'Hello World')
    assert(text_no_tags[3] == '5 > 4 and 4 < 3')
    assert(text_no_tags[4] == '4  3')
    
def test_remove_emojis():
    '''
    This test function tests the correct working of the remove_emojis function.
    '''
    text_with_emojis = ['Hello world \U0001f600',
                        'This is a dog: \U0001F436',
                        'This is a rainbow: \U0001F308',
                        '\U0000274C this is a cross mark',
                        'This is the number three (3): \U00000033',
                        'This is the letter B: \U0001F171',
                        'This is SOS: \U0001F198',
                        'This is red circle: \U0001F534']
    
    text_with_no_emojis = []
    for text in text_with_emojis:
        text_no_emoji = remove_emojis(text)
        text_with_no_emojis.append(text_no_emoji)

    assert(text_with_no_emojis[0] == 'Hello world ')
    assert(text_with_no_emojis[1] == 'This is a dog: ')
    assert(text_with_no_emojis[2] == 'This is a rainbow: ')
    assert(text_with_no_emojis[3] == ' this is a cross mark')
    assert(text_with_no_emojis[4] == 'This is the number three (3): 3')
    assert(text_with_no_emojis[5] == 'This is the letter B: ')
    assert(text_with_no_emojis[6] == 'This is SOS: ')
    assert(text_with_no_emojis[7] == 'This is red circle: ')

#TODO: PROBABILMENTE LE DUE FUNZIONI SCRITTE QUI SOTTO SONO DI TROPPO

'''
def test_remove_quotation_marks():
    '''
#    This test function tests the correct working of the remove_quotmarks_underscore function.
#    In particular, this test functions tests that the quotation marks or the apostrophes are eliminated from the sentences
#    with leaving one space between words. In this way, if we have something like "I've done something...", 
#    it remains just "I ve done something..." and with the lemmatizer and the stopwords it's possible to remove both
#    "I" and "ve".
'''
    text_example1 = "Hello world, it's me"
    text_no_quot_mark1 = remove_quotmarks_underscore(text_example1)

    text_example2 = '"Hello world" he said'
    text_no_quot_mark2 = remove_quotmarks_underscore(text_example2)

    assert(text_no_quot_mark1 == 'Hello world, it s me')
    assert(text_no_quot_mark2 == ' Hello world  he said')

'''

'''
def test_rm_underscore():
'''
#    This test function tests the correct working of the remove_quotmarks_underscore function.
#    In particular, this test functions tests that underscores are eliminated from the sentences
#    with leaving one spaces between words.
'''
    text_example1 = "__Hello world__"
    text_no_underscore1 = remove_quotmarks_underscore(text_example1)

    text_example2 = 'Hello_world'
    text_no_underscore2 = remove_quotmarks_underscore(text_example2)

    assert(text_no_underscore1 == ' Hello world ')
    assert(text_no_underscore2 == 'Hello world')
'''

def test_rm_noalphanumeric():
    '''
    This test function tests the correct working of the 
    remove_noalphanum_singlechar function.
    In particular, this test functions tests that all the non alphanumeric
    characters are removed from the text.
    '''
    test_allchars = ["Hashtag try: #hello world!!",
                     "random.email@gmail.com",
                     "<TEXT> html tag",
                     "wikipedia site: https://en.wikipedia.org/wiki",
                     "try with underscore:____",
                     "try with some numbers: 3092"]
    
    text_noalphanumeric = []
    for allchars in test_allchars:
        noalphanumeric = remove_noalphanum(allchars)
        text_noalphanumeric.append(noalphanumeric)

    assert(text_noalphanumeric[0] == 'Hashtag try   hello world  ')
    assert(text_noalphanumeric[1] == 'random email gmail com')
    assert(text_noalphanumeric[2] == " TEXT  html tag")
    assert(text_noalphanumeric[3] == "wikipedia site  https   en wikipedia org wiki")
    assert(text_noalphanumeric[4] == "try with underscore     ")
    assert(text_noalphanumeric[5] == "try with some numbers  3092")


# TODO: DA DECIDERE SE TENERE O NO LA FUNZIONE QUI SOTTO..
def test_rm_singlechar():
    '''
    This test function tests the correct working of the 
    remove_noalphanum_singlechar function.
    In particular, this test functions tests that all the single
    characters are removed from the text (the 's' after the apostrophe for example.)
    '''
    test_example = "Hello, it's me"
    test_no_single_char = remove_noalphanum(test_example)

    test_example1 = "I've done something"
    test_twochars = remove_noalphanum(test_example1)

    assert(test_no_single_char == 'Hello  it me')
    assert(test_twochars == "I ve done something")


def test_clean_text():
    '''
    This test functions test the correct working of the clean_text function.
    The idea is to remove all the punctuations, all the non alphanumeric characters,
    URLs, HTML tags, apostrophes, quotation marks and underscore
    '''
    text_to_clean = ['Hello, my name is Flavio!',
                     '<SITE> wikpedia site: https://en.wikipedia.org/wiki',
                     '#project for binary classification.',
                     "That's my car: a very nice one!",
                     'User_ 12: "Hello my name is user_12"']
    
    texts_cleaned = []
    for text in text_to_clean:
        cleaned_text = clean_text(text)
        texts_cleaned.append(cleaned_text)

    print(texts_cleaned[4])
    assert(texts_cleaned[0] == 'hello my name is flavio')
    assert(texts_cleaned[1] == 'wikpedia site')
    assert(texts_cleaned[2] == 'project for binary classification')
    assert(texts_cleaned[3] == 'that s my car a very nice one')
    assert(texts_cleaned[4] == 'user 12 hello my name is user 12')


def test_stopword():
    '''
    This test functions test the correct working of the stopwrod function.
    The idea is to remove all the stop words in the english vocabulary.
    The list for the all stop words can be seen with these three lines of code.
    
    import nltk
    from nltk.corpus import stopwords
    print(stopwords.words('english')) 
    '''
    sentences = ['I have many pets at home.',
                "That's my pen!",
                'Hello world, how are you?',
                'A random sentence to see how it works.']

    sentences_stop_word = []
    for text_example in sentences:
        test_clean_text = clean_text(text_example)
        test_no_stopword = stopword(test_clean_text)
        sentences_stop_word.append(test_no_stopword)

    
    assert(sentences_stop_word[0] == 'many pets home')
    assert(sentences_stop_word[1] == 'pen')
    assert(sentences_stop_word[2] == 'hello world')
    assert(sentences_stop_word[3] == 'random sentence see works')
