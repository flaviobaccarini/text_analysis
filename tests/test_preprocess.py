'''
Test functions for testing the functions inside the clean_text module.
'''
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
import string
import random 
from hypothesis import strategies as st
from hypothesis import given
from binary_classifier.cleantext import lower_strip, finalpreprocess
from binary_classifier.cleantext import remove_urls_tags, get_wordnet_pos
from binary_classifier.cleantext import remove_noalphanum, stopword
from binary_classifier.cleantext import clean_text, lemmatizer
from binary_classifier.cleantext import rename_columns, drop_empty_rows

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
                     "try with some numbers: 3092",
                     "emoticons: \U0000274C\U0001f600\U0001F436\U0001F534"]
    
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
    assert(text_noalphanumeric[6] == "emoticons      ")

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
                     'User_ 12: "Hello my name is user_12"',
                     '\U0000274C this is a cross mark',
                     'This is the number three (3): \U00000033']
    
    texts_cleaned = []
    for text in text_to_clean:
        cleaned_text = clean_text(text)
        texts_cleaned.append(cleaned_text)

    assert(texts_cleaned[0] == 'hello my name is flavio')
    assert(texts_cleaned[1] == 'wikpedia site')
    assert(texts_cleaned[2] == 'project for binary classification')
    assert(texts_cleaned[3] == 'that s my car a very nice one')
    assert(texts_cleaned[4] == 'user 12 hello my name is user 12')
    assert(texts_cleaned[5] == 'this is a cross mark')
    assert(texts_cleaned[6] == 'this is the number three 3 3')

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


def test_lemmatizer():
    '''
    Test function to test the correct working for the lemmatizer function.
    The lemmatizer function take a text (string) as input and lemmatize the text.
    The output is the lemmatized text.
    '''
    texts_to_lemmatize = ["The striped bats are hanging on their feet for best",
                          "All the players are playing.",
                          "Cats and dogs usually are not best friends.",
                          "Accordingly to the weather forecast tomorrow it is going to rain",
                          "the text at this preprocess point would be all lower and without punctuations",
                        ]
    texts_lemmatized = []

    # let's see how all the process works: 
    # 1) we start from a lower text with no punctuation 
    # 2) we apply the stopword
    # 3) we lemmatize the text 
    texts_to_lemmatize[4] = stopword(texts_to_lemmatize[4])

    for text in texts_to_lemmatize:
        lemmatized_text = lemmatizer(text)
        texts_lemmatized.append(lemmatized_text)

    assert(texts_lemmatized[0] == 'The striped bat be hang on their foot for best')
    assert(texts_lemmatized[1] == 'All the player be play .')
    assert(texts_lemmatized[2] == 'Cats and dog usually be not best friend .')
    assert(texts_lemmatized[3] == "Accordingly to the weather forecast tomorrow it be go to rain")
    assert(texts_lemmatized[4] == 'text preprocess point would lower without punctuation')

def test_get_wordnet_pos():
    '''
    Test function to test how the get_wordnet_pos function works.
    The idea is to match all the words from the sentences with the wordnet tags,
    in order to lemmatize the words after this process.
    '''
    text_to_test = ['This is a try',
                    "it is incredibly beautiful",
                    "i run outside if the weather is really good"]
    tags = [nltk.pos_tag(word_tokenize(text)) for text in text_to_test]

    wordnet_pos_tags = []
    n = 1
    for tag in tags:
        only_tag = [x[n] for x in tag]
        wordnet_pos = [get_wordnet_pos(tag) for tag in only_tag]
        wordnet_pos_tags.append(wordnet_pos)
  
    assert(wordnet_pos_tags[0] == ['n', 'v', 'n', 'n']) # NOUN, VERB, NOUN, NOUN
    assert(wordnet_pos_tags[1] == ['n', 'v', 'r', 'a']) # NOUN, VERB, ADVERB, ADJECTIVE
    assert(wordnet_pos_tags[2] == ['n', 'v', 'n', 'n', 'n', 'n', 'v', 'r', 'a'])
    


def test_final_preprocess():
    '''
    Test function to test how the finalpreprocess function works.
    Final preprocess is the function that applies all the functions
    for cleaning the text. 
    The output of this function should be only some meaningful words
    for the sentence to analyze.
    '''

    text_example = ['Some random text to use',
                    'This year 2022 is fantastic',
                    "I don't know what to do!",
                    "this is a try with a tag: #TAG",
                    '"Hello this is my website: https://wikipedia.org!"',
                    "<TEXT> here there is some text <TEXT>",
                    "This is my favorite astronaut \U0001F9D1!",
                    "The parents are watching the TV",
                    "How are you?",
                    "@User_10 what are you doing?",
                    "The rainbow \U0001F308 is very beautiful"]

    text_processed = []

    for text in text_example:
        text_processed.append(finalpreprocess(text))

    assert(text_processed[0] == 'random text use')
    assert(text_processed[1] == 'year 2022 fantastic')
    assert(text_processed[2] == 'know')
    assert(text_processed[3] == 'try tag tag')
    assert(text_processed[4] == 'hello website')
    assert(text_processed[5] == 'text')
    assert(text_processed[6] == 'favorite astronaut')
    assert(text_processed[7] == 'parent watch tv')
    assert(text_processed[8] == '')
    assert(text_processed[9] == 'user 10')
    assert(text_processed[10] == 'rainbow beautiful')


@given(column_names = st.lists(st.text(max_size=10),
                                 min_size=2, max_size=2, unique = True)) 
def test_rename_columns(column_names):
    '''
    Test function to test the working of the rename_columns function.
    The rename_columns function takes as input the dataframe that the user wants
    to change the column names and the text's and labels' column names. 
    The order of the column names given as input to the function is important:
    so the list (or tuple) must be always composed by the first string that corresponds
    to the text's column name and the second string, which corresponds to the label's column name.
    The output is a new dataframe composed by just two columns with the names: "text", "label".
    '''

    unique_labels = ['real', 'fake']

    sentences = [' '.join(random.choices(string.ascii_letters + 
                             string.digits, k=10)) for _ in range(100)]
    labels = [unique_labels[random.randrange(len(unique_labels))] for _ in range(100)]

    other_text = [' '.join(random.choices(string.ascii_letters + 
                             string.digits, k=10)) for _ in range(100)]

    df_fake = pd.DataFrame({column_names[0]: sentences,
                            column_names[1]: labels, 'other': other_text })
    
    df_correct_col_names = rename_columns(df_fake, column_names)

    assert(len(df_correct_col_names.columns.values.tolist()) == 2) # only two columns
    assert(df_correct_col_names.columns.values.tolist() == ['text', 'label']) # check name of colums are correct
    if len(df_correct_col_names) != 0: # check if in the label column there are actually the labels
        assert(list(df_correct_col_names['label']).count('real') > 0 or
                             list(df_correct_col_names['label']).count('fake') > 0 )

    # THE ORDER OF THE COLUMNS IN THE DATAFRAME IS NOT IMPORTANT
    # THE IMPORTANT IS THE ORDER OF THE COLUMN NAMES IN THE rename_columns FUNCTION
    df_fake_inverted_order = pd.DataFrame({column_names[1]: labels,
                                           column_names[0]: sentences})
    df_inverted_order_colnames = rename_columns(df_fake_inverted_order, column_names)
    assert(df_inverted_order_colnames.columns.values.tolist() == ['text', 'label']) # check name of colums are correct
    if len(df_inverted_order_colnames) != 0: # check if in the label column there are actually the labels
        assert(list(df_inverted_order_colnames['label']).count('real') > 0 or 
                    list(df_inverted_order_colnames['label']).count('fake') > 0 )


@given(all_sentences = st.lists(st.text(min_size=0, max_size=20),
                                        min_size = 0, max_size = 20))
def test_drop_empty(all_sentences):
    '''
    Test function to prove the correctly dropping of the empty rows in the dataframe.
    The initial dataset can be composed of empty cells, or cells with no text inside (only '').
    This kind of data are not so meaningful, so this function cares to remove these rows.
    The final dataset contain only rows with text.
    '''
    unique_labels = ['real', 'fake',]
    labels = [unique_labels[random.randrange(len(unique_labels))] 
                                            for _ in range(len(all_sentences))]

    df_fake = pd.DataFrame({'text': all_sentences, 'label': labels})

    df_no_empty_rows = drop_empty_rows(df_fake)

    assert( len(np.where(pd.isnull(df_no_empty_rows))[0]) == 0 ) # no empty cells for text
    assert( len(np.where(pd.isnull(df_no_empty_rows))[1]) == 0 ) # no empty cells for label
    assert( len(np.where(df_no_empty_rows.applymap(lambda x: x == ''))[0]) == 0 ) # no cells '' for text
    assert( len(np.where(df_no_empty_rows.applymap(lambda x: x == ''))[1]) == 0 ) # no cells '' for label

    if len(df_no_empty_rows )> 0:
        text_count = map(len, df_no_empty_rows['text'])
        assert(min(text_count) >= 1)

