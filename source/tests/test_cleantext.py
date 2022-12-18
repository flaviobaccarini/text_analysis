'''
Test functions for testing the functions inside the cleantext module.
'''
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from hypothesis import strategies as st
from hypothesis import given
from text_analysis.cleantext import lower_strip, finalpreprocess
from text_analysis.cleantext import remove_urls_tags, get_wordnet_pos
from text_analysis.cleantext import remove_noalphanum, stopword
from text_analysis.cleantext import clean_text, lemmatizer
from text_analysis.cleantext import rename_columns, drop_empty_rows


def test_lower_all_capital_letters():
    '''
    This test function tests the behaviour of lower_strip function.
    In particular, in this test function the lower part is tested 
    starting from a text composed only by capital letters. The output should
    be the same text with all lowercase letters.

    Given:
    =======
    capital_text: string with all capital letters.
    
    Tests:
    =======
          The output string is composed by only lowercase letters.
    '''
    capital_text = 'HELLO WORLD'
    text_lower = lower_strip(capital_text)
    assert(text_lower == 'hello world')

def test_lower_some_capital_letters():
    '''
    This test function tests the behaviour of lower_strip function.
    In particular, in this test function the lower part is tested 
    starting from a text with som capital letters within the text.
    The output should be the same text with all lowercase letters.

    Given:
    =======
    capital_text: string with some capital letters in the text.
    
    Tests:
    =======
          The output string is composed by only lowercase letters.
    '''
    capital_text = 'HellO WoRld'
    text_lower = lower_strip(capital_text)
    assert(text_lower == 'hello world')

def test_lower_some_capital_letters():
    '''
    This test function tests the behaviour of lower_strip function.
    In particular, in this test function the lower part is tested 
    starting from a text with only lowercase letters.
    The output should be the same text (no changes).

    Given:
    =======
    lowercase_text: string with only lowercase letters.
    
    Tests:
    =======
          The output string is composed by only lowercase letters.
          In this case the output string is the same as the input.
    '''
    lowercase_text = 'hello world'
    text_lower = lower_strip(lowercase_text)
    assert(text_lower == 'hello world')


def test_rm_initial_whitespaces():
    '''
    This test function tests the behaviour of lower_strip function.
    In particular, in this test function the stripping part is tested
    starting from a text with some whitespaces at the begininning of
    the text.
    
    Given:
    =======
    text_with_whitespaces: str
                           Initial text with initial whitespaces
    
    Tests:
    ======
         The output text should be the original text without the initial
         whitespaces.
    '''
    text_with_whitespaces = '      hello world'
    text_no_whitespaces = lower_strip(text_with_whitespaces)
    assert(text_no_whitespaces == 'hello world')


def test_rm_middle_whitespaces():
    '''
    This test function tests the behaviour of lower_strip function.
    In particular, in this test function the stripping part is tested
    starting from a text with some whitespaces in the middle of the 
    text.
    
    Given:
    =======
    text_with_whitespaces: str
                           Initial text with whitespaces in the middle
                           of the text
    
    Tests:
    ======
         The output text should be the original text without the big
         whitespace in the middle (only a single whitespace: ' ').
    '''
    text_with_whitespaces = 'hello         world' 
    text_no_whitespaces = lower_strip(text_with_whitespaces)
    assert(text_no_whitespaces == 'hello world')


def test_rm_final_whitespaces():
    '''
    This test function tests the behaviour of lower_strip function.
    In particular, in this test function the stripping part is tested
    starting from a text with some whitespaces at the end of the 
    text.
    
    Given:
    =======
    text_with_whitespaces: str
                           Initial text with whitespaces at the end
                           of the text
    
    Tests:
    ======
         The output text should be the original text without the 
         whitespaces at the end of the text.
    '''
    text_with_whitespaces = 'hello world            '
    text_no_whitespaces = lower_strip(text_with_whitespaces)
    assert(text_no_whitespaces == 'hello world')

# TODO: PROBABILMENTE DA ELIMINARE
@given(text = st.text(min_size = 0, max_size = 20))
def test_lower_strip(text):
    '''
    This test function tests the behaviour of the lower_strip function
    with some random text given as input.
    
    @given:
    ========
    text: str
          Random generated text.
    '''
    # 1) lower the text; 2) strip the text; 3) remove whitespaces
    correct_text = " ".join(text.lower().split())
    text_from_function = lower_strip(text)
    assert(text_from_function == correct_text)

def test_rm_final_url():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In particular, in this test function only the part about URLs is tested.
    The URL is at the end of the text.

    Given:
    ======
    text_with_url: str
                   Text with URL at the end of the text.
    
    Tests:
    ======
         The final text after the function doesn't include the URL at
         the end.
    '''
    text_with_url = 'This is wikipedia site: https://en.wikipedia.org/wiki'
    text_without_url =  remove_urls_tags(text_with_url)
    assert(text_without_url == 'This is wikipedia site: ')

def test_rm_initial_url():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In particular, in this test function only the part about URLs is tested.
    The URL is at the beginning of the text.

    Given:
    ======
    text_with_url: str
                   Text with URL at the begininning of the text.
    
    Tests:
    ======
         The final text after the function doesn't include the URL at
         the beninning.
    '''
    text_with_url = 'https://en.wikipedia.org/wiki this is wikipedia site'
    text_without_url =  remove_urls_tags(text_with_url)
    print(text_without_url)
    assert(text_without_url == ' this is wikipedia site')


def test_rm_middle_url():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In particular, in this test function only the part about URLs is tested.
    The URL is in the middle of the text.

    Given:
    ======
    text_with_url: str
                   Text with URL at the end of the text.
    
    Tests:
    ======
         The final text after the function doesn't include the URL at
         the end.
    '''
    text_with_url = 'The wikipedia site https://en.wikipedia.org/wiki is here'
    text_without_url =  remove_urls_tags(text_with_url)
    assert(text_without_url == 'The wikipedia site  is here')


def test_rm_no_tags_or_url():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In this test function the text doesn't contain any tags or URLs,
    so it should not change after the function,

    Given:
    ======
    text_no_tag_or_url: str
                        Text with no tags or URLs. 
    
    Tests:
    ======
         The final text should be equal to the input text, since
         it does not contain any URLs or tags.
    '''
    text_no_tag_or_url = 'Hello World'
    text_without_url =  remove_urls_tags(text_no_tag_or_url)
    assert(text_without_url == 'Hello World')

def test_rm_only_url():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In this test function the input text contains only a URL.

    Given:
    ======
    url_link: str
              URL link.   

    Tests:
    ======
         The final string should be an empty string, since in the input
         string there is only a URL.
    '''
    url_link = 'https://en.wikipedia.org/wiki'
    text_without_url =  remove_urls_tags(url_link)
    assert(text_without_url == '')    

def test_remove_initial_tag():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In particular, in this test function only the part about tags is tested
    starting from a text with a tag at the beginning of the text.

    Given:
    ======
    text_with_tag: str
                   Text with tag at the begininning.

    Tests:
    =======
            The output text should be the text without the
            tag at the beginning.
    '''
    text_with_tag = '<TEXT> Hello World'
    text_without_tag = remove_urls_tags(text_with_tag)
    assert(text_without_tag== ' Hello World')

def test_remove_tag_and_url():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In particular, in this test function it's tested what's the result
    if a string is composed only by a tag and a URL.

    Given:
    ======
    text_with_tag_and_url: str
                           Text composed only by a tag and an URL.

    Tests:
    =======
            The output text should be an empty string composed only
            by whitespaces.
    '''
    text_with_tag_and_url = '<TEXT> http://wikipedia.com'
    text_without_tag_and_url = remove_urls_tags(text_with_tag_and_url)
    assert(text_without_tag_and_url== ' ')


def test_remove_final_tag():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In particular, in this test function only the part about tags is tested
    starting from a text with a tag at the end of the text.

    Given:
    ======
    text_with_tag: str
                   Text with tag at the end.

    Tests:
    =======
            The output text should be the text without the
            tag at the end.
    '''
    text_with_tag = 'Hello World <TEXT>'
    text_without_tag = remove_urls_tags(text_with_tag)
    assert(text_without_tag== 'Hello World ')

def test_remove_middle_tag():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In particular, in this test function only the part about tags is tested
    starting from a text with a tag in the middle of the text.

    Given:
    ======
    text_with_tag: str
                   Text with tag within the text.

    Tests:
    =======
            The output text should be the text without the
            tag in the middle.
    '''
    text_with_tag = 'Hello <TEXT> World'
    text_without_tag = remove_urls_tags(text_with_tag)
    assert(text_without_tag== 'Hello  World')


def test_remove_only_tag():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In particular, in this test function only the part about tags is tested
    starting from a string with only a tag.

    Given:
    ======
    tag: str
         A single tag.

    Tests:
    =======
         The output string should be the an empty string.
    '''
    tag = '<TEXT>'
    text_without_tag = remove_urls_tags(tag)
    assert(text_without_tag== '')

def test_remove_two_tags():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In particular, in this test function only the part about tags is tested
    starting from a text with two tags: one at the beginning and one 
    at the end of the text.

    Given:
    ======
    text_with_tags: str
                    A string containing two tags.

    Tests:
    =======
            The output string should be the initial text without the 
            two tags..
    '''
    text_with_tags = '<TEXT> Hello World <TEXT>'
    text_without_tag = remove_urls_tags(text_with_tags)
    assert(text_without_tag == ' Hello World ')

def test_rm_punctuations():
    '''
    This test function tests the behaviour of remove_noalphanum function.
    In particular, this test function tests that all the punctuations are eliminated 
    from the initial string.

    Given:
    ======
    text_with_punctuations: str
                            Text string with punctuations.

    Tests:
    ======
          The output string should not contain any punctuations.
    '''
    text_with_punctuations = "Hello world, some punctuations!?"
    text_no_punct = remove_noalphanum(text_with_punctuations)
    assert(text_no_punct == 'Hello world  some punctuations  ')


def test_rm_emoticons():
    '''
    This test function tests the behaviour of remove_noalphanum function.
    In particular, this test function tests that emoticons are eliminated 
    from the initial text.

    Given:
    ======
    text_with_emoticons: str
                         Text string with emoticons.

    Tests:
    ======
          The output string should not contain any emoticons.
    '''
    text_with_emoticons = "emoticons \U0000274C\U0001f600\U0001F436\U0001F534"
    text_no_emoticons = remove_noalphanum(text_with_emoticons)
    assert(text_no_emoticons == "emoticons     ")


def test_rm_specialchars():
    '''
    This test function tests the behaviour of remove_noalphanum function.
    In particular, this test function tests that special characters are eliminated 
    from the initial text (special characters: #, @ etc...)

    Given:
    ======
    text_with_spec_chars: str
                          Text string with special characters.

    Tests:
    ======
          The output string should not contain any special characters 
          (instead of the special characters we will find whitespaces).
    '''
    text_with_spec_chars = "special chars #@><-*+/"
    text_no_spec_chars = remove_noalphanum(text_with_spec_chars)
    assert(text_no_spec_chars == 'special chars         ')

def test_rm_underscores():
    '''
    This test function tests the behaviour of remove_noalphanum function.
    In particular, this test function tests that underscores are eliminated 
    from the initial text.

    Given:
    ======
    text_with_underscore: str
                          Text string with underscores.

    Tests:
    ======
          The output string should not contain any underscores.
    '''
    text_with_underscores = "test __"
    text_no_underscores = remove_noalphanum(text_with_underscores)
    assert(text_no_underscores == "test   ")

def test_rm_quot_marks():
    '''
    This test function tests the behaviour of remove_noalphanum function.
    In particular, this test function tests that quotation marks are eliminated 
    from the initial text.

    Given:
    ======
    text_with_quot_makrs: str
                          Text string with quotation marks.

    Tests:
    ======
          The output string should not contain any quotation marks.
    '''
    text_with_quot_marks = 'try with "hello"'
    text_no_quot_marks = remove_noalphanum(text_with_quot_marks)
    assert(text_no_quot_marks == "try with  hello ")

def test_rm_apostrophe():
    '''
    This test function tests the behaviour of remove_noalphanum function.
    In particular, this test function tests that apostrophes are eliminated 
    from the initial text.

    Given:
    ======
    text_with_apostrophe: str
                          Text string with apostrophe.

    Tests:
    ======
          The output string should not contain any apostrophe.
    '''
    text_with_aposrophe = "i'm flavio let's try"
    text_no_apostrophe = remove_noalphanum(text_with_aposrophe)
    assert(text_no_apostrophe == "i m flavio let s try")

def test_clean_url():
    '''
    Test function to test behaviour of clean_text function.
    In this test function it is tested  what's the result if
    inside the text there is a URL.

    Given:
    ======
    text_with_url: str
                   Text with url.
    
    Tests:
    ======
            The output string should be a text without URL.
    '''
    text_with_url = "wikipedia site https://en.wikipedia.org/wiki"
    text_no_url = clean_text(text_with_url)
    assert(text_no_url == "wikipedia site")

    
def test_clean_tag():
    '''
    Test function to test behaviour of clean_text function.
    In this test function it is tested what's the result 
    if inside the text there is a tag.

    Given:
    ======
    text_with_tag: str
                   Text with tag.
    
    Tests:
    ======
            The output string should be a text without tag.
    '''
    text_with_tag = "<NAME> flavio <NAME>"
    text_no_tag = clean_text(text_with_tag)
    assert(text_no_tag == "flavio")
    
def test_clean_punct():
    '''
    Test function to test behaviour of clean_text function.
    In this test function it is tested what's the results
    if inside the text there are punctuations.

    Given:
    ======
    text_with_punct: str
                     Text with punctuations.
    
    Tests:
    ======
            The output string should be a text without punctuations.
    '''
    text_with_punct = "hello, my name is flavio!"
    text_no_punct = clean_text(text_with_punct)
    assert(text_no_punct == "hello my name is flavio")
    
  
def test_clean_whitespaces():
    '''
    Test function to test behaviour of clean_text function.
    In this test function it is tested what's the results
    if inside the text there are big whitespaces.

    Given:
    ======
    text_with_whitespaces: str
                           Text with big whitespaces.
    
    Tests:
    ======
            The output string should be a text containing only
            single whitespaces (only ' ').
    '''
    text_with_whitespaces = "hello    world  my   name    is    flavio"
    text_no_whitespaces = clean_text(text_with_whitespaces)
    assert(text_no_whitespaces == "hello world my name is flavio")
    
def test_clean_capital_letters():
    '''
    Test function to test behaviour of clean_text function.
    In this test function it is tested what's the results
    if inside the text there are big capital letters.

    Given:
    ======
    text_with_capitals: str
                        Text with capital letters.
    
    Tests:
    ======
            The output string should be a text containing only
            lowercase letters.
    '''
    text_with_capitals = "Hello World MY NAME is FLAviO "
    text_no_capitals = clean_text(text_with_capitals)
    assert(text_no_capitals == "hello world my name is flavio")
    

def test_clean_specialchars():
    '''
    Test function to test behaviour of clean_text function.
    In this test function it is tested what's the results
    if inside the text there are big special characters.

    Given:
    ======
    text_with_specialchars: str
                            Text with special characters.
    
    Tests:
    ======
            The output string should be a text without special 
            characters.
    '''
    text_with_specialchars = "#hello #world my name is @flavio"
    text_no_specialchars = clean_text(text_with_specialchars)
    assert(text_no_specialchars == "hello world my name is flavio")


def test_clean_emoticons_and_cap_letters():
    '''
    Test function to test behaviour of clean_text function.
    In this test function it is tested what's the results
    if inside the text there are emoticons and capital letters.

    Given:
    ======
    text_with_emoticons_and_cap_letters: str
                                         Text with emoticons and capital letters.
    
    Tests:
    ======
            The output string should be a text without emoticons and 
            wiht only lowercase letters.
    '''
    text_with_emoticons_and_cap_letters = '\U0000274C This is a Cross Mark'
    text_no_emoticons_and_cap_letters = clean_text(text_with_emoticons_and_cap_letters)
    assert(text_no_emoticons_and_cap_letters == "this is a cross mark")


def test_stopword():
    '''
    This test function tests the behaviour of stopword function.
    The idea is to remove all the words present in the stop list 
    for the english vocabulary, because they are meaningless.
    The list of the all stop words can be seen with these lines of code.
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    print(stopwords.words('english')) 

    Given:
    ======
    text_with_stopwords: str
                         Text with english stop words.
    
    Tests:
    ======
            The output string should be a text without stop words.
    '''
    text_with_stopwords = "a random sentence to see how it works"
    text_no_stopword = stopword(text_with_stopwords)
    assert(text_no_stopword == 'random sentence see works')

def test_no_stopwords_in_text():
    '''
    This test function tests the behaviour of stopword function.
    The idea is to remove all the words present in the stop list 
    for the english vocabulary, because they are meaningless.
    The list of the all stop words can be seen with these lines of code.
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    print(stopwords.words('english')) 

    In this test function it's tested what's the result if within the 
    input text there are no stop words.

    Given:
    ======
    text_without_stopwords: str
                            Text without stop words.
    
    Tests:
    ======
            The output string should be equal to the input text, since
            it doesn't contain any stop word.
    '''
    text_without_stopwords = "random sentence without stop words"
    text_no_stopword = stopword(text_without_stopwords)
    assert(text_no_stopword == 'random sentence without stop words')


def test_only_stopwords_in_text():
    '''
    This test function tests the behaviour of stopword function.
    The idea is to remove all the words present in the stop list 
    for the english vocabulary, because they are meaningless.
    The list of the all stop words can be seen with these lines of code.
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    print(stopwords.words('english')) 

    In this test function it's tested what's the result if within the 
    input text there are only stop words.

    Given:
    ======
    text_with_only_stopwords: str
                              Text composed by only stop words.
    
    Tests:
    ======
            The output string should be an empty string (""), since
            the original string is composed only by stop words.
    '''
    text_with_only_stopwords = "how are you"
    text_no_stopword = stopword(text_with_only_stopwords)
    assert(text_no_stopword == '')

def test_lemmatizer():
    '''
    Test function to test the behaviour of lemmatizer function.
    The lemmatizer function takes a text (string) as input and lemmatize the text.
    The output is the lemmatized text.

    Given:
    ======
    text_to_lemmatize: str
                       Text string that needs to be lemmatized.
    
    Tests:
    ======
            The output text should be the lemmatized text.
    '''
    text_to_lemmatize = "the striped bats are hanging on their feet for best"
    lemmatized_text = lemmatizer(text_to_lemmatize)
    assert(lemmatized_text == 'the striped bat be hang on their foot for best')

def test_lemmatizer_text_already_lemmatized():
    '''
    Test function to test the behaviour of lemmatizer function.
    The lemmatizer function takes a text (string) as input and lemmatize the text.
    This test function tests what's the result if the input text to lemmatizer 
    is already lemmatized.    

    Given:
    ======
    text_already_lemmatized: str
                             Text string already lemmatized.
    
    Tests:
    ======
            The output text should be equal to the input text string
            since the original string is already lemmatized.
    '''
    text_already_lemmatized = "I play basketball in my free time"
    lemmatized_text = lemmatizer(text_already_lemmatized)
    assert(lemmatized_text == "I play basketball in my free time")


def test_get_wordnet_pos():
    '''
    Test function to test the behaviour of get_wordnet_pos function.
    The idea is to match all the words from the sentences with the wordnet tags,
    in order to lemmatize correctly the words after this process.
    In this way the lemmatizer understand if the word is a noun, verb, adverb
    or adjective.

    Given:
    ======
    text_for_tag: str
                  Text to extrapolate wordnet tags.
    
    Tests:
    ======
            The output should be a string that represents if 
            the word is a noun, verb, adjective or adverb.
            Noun = 'n', verb = 'v', adverb = 'r', adjective = 'j'
    '''
    text_for_tag = "it is incredibly beautiful"
    tags = nltk.pos_tag(word_tokenize(text_for_tag))
    wordnet_pos = [get_wordnet_pos(tag[1]) for tag in tags]
    assert(wordnet_pos == ['n', 'v', 'r', 'a']) # NOUN, VERB, ADVERB, ADJECTIVE


# TODO: scegliere cosa fare di questa funzione
def test_final_preprocess():
    '''
    Test function to test the behaviour of finalpreprocess function.
    finalpreprocess is the function that applies all the functions
    for cleaning the text (clean_text, stopword and lemmatize)
    The output of this function should be only some meaningful lemmatized words
    for the text.
    '''
    # some random text to preprocess
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
    # preprocessed text
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


def test_rename_columns():
    '''
    Test function to test the behaviour of rename_columns function.
    The rename_columns function takes as input the dataframe that the user wants
    to change the column names and the text's and labels' original column names. 
    The output is a new dataframe composed by just two columns with the names:
    "text", "label".

    Given:
    =======
    dataframe_init_columns: pd.DataFrame
                            Oriiginal dataframe with initial column names.

    Tests:
    ======
            This test function tests that in the output dataframe there are
            the columns named: 'text', 'label'
    '''
    dataframe_init_column = pd.DataFrame(columns=('original_text', 'original_target'))
    df_correct_col_names = rename_columns(dataframe_init_column,
                                          text_column_name = 'original_text', 
                                          label_column_name = 'original_target')
    assert(df_correct_col_names.columns.values.tolist() == ['text', 'label']) 


def test_rename_columns_number_of_columns():
    '''
    Test function to test the behaviour of rename_columns function.
    The rename_columns function takes as input the dataframe that the user wants
    to change the column names and the text's and labels' original column names. 
    In this test function it is tested that there are only two columns in the 
    output datafarme.

    Given:
    =======
    dataframe_init_columns: pd.DataFrame
                            Oriiginal dataframe with initial column names.

    Tests:
    ======
            This test function tests that in the output dataframe there are
            only two columns.
    '''
    dataframe_init_column = pd.DataFrame(columns=('original_text', 'original_target',
                                                  'new_col', 'new_column2'))
    df_correct_col_names = rename_columns(dataframe_init_column,
                                          text_column_name = 'original_text', 
                                          label_column_name = 'original_target')
    assert(len(df_correct_col_names.columns.values.tolist()) == 2) 
    
def test_rename_columns_inverted_order():
    '''
    Test function to test the behaviour of rename_columns function.
    The rename_columns function takes as input the dataframe that the user wants
    to change the column names and the text's and labels' original column names. 
    In this test function it is tested that the output dataframe contains two columns
    ('label' and 'text') also if in the original dataframe there is first the label 
    column than the text column.

    Given:
    =======
    dataframe_init_columns: pd.DataFrame
                            Oriiginal dataframe with initial column names.

    Tests:
    ======
            This test function tests that in the output dataframe there are
            'text' and 'label' column also if in the original dataframe the 
            order is first the label column, then the text column.
    '''  
    dataframe_init_column = pd.DataFrame(columns=('original_text', 'original_target'))
    df_correct_col_names = rename_columns(dataframe_init_column,
                                          text_column_name = 'original_text', 
                                          label_column_name = 'original_target')
    assert(df_correct_col_names.columns.values.tolist() == ['text', 'label']) 

def test_drop_empty_for_text():
    '''
    Test function to test the behaviour of drop_empty_rows function.
    The initial dataset can be composed of NaN/None cells, or cells with
    no text inside (only ''). This kind of data are not so meaningful,
    so drop_empty_rows cares to remove these rows.
    The final dataset contains only rows with text.

    In this test function it is tested that there are no empty cells 
    for the 'text' column, starting from a dataframe with '' cells in 
    the 'text' column.

    Given:
    ========
    init_dataframe: pd.DataFrame
                    Original dataframe with empty cells in 'text' column.

    Tests:
    ======
            The output dataframe should not contain any empty cell for the
            'text' column.
    '''
    init_dataframe = pd.DataFrame({'text': ['a', '', 'c'], 'label':[0,0,1]})
    df_no_empty_cells_text = drop_empty_rows(init_dataframe)
    # no cells '' for text:
    assert( len(np.where(df_no_empty_cells_text.applymap(lambda x: x == ''))[0]) == 0 ) 

def test_drop_empty_for_label():
    '''
    Test function to test the behaviour of drop_empty_rows function.
    The initial dataset can be composed of NaN/None cells, or cells with
    no text inside (only ''). This kind of data are not so meaningful,
    so drop_empty_rows cares to remove these rows.
    The final dataset contains only rows with text.

    In this test function it is tested that there are no empty cells 
    for the 'label' column, starting from a dataframe with '' cells in 
    the 'label' column.

    Given:
    ========
    init_dataframe: pd.DataFrame
                    Original dataframe with empty cells in 'label' column.

    Tests:
    ======
            The output dataframe should not contain any empty cell for the
            'label' column.
    '''
    init_dataframe = pd.DataFrame({'text': ['a', 'b', 'c'], 'label':['0','','1']})
    df_no_empty_cells_label = drop_empty_rows(init_dataframe)
    # no cells '' for label:
    assert( len(np.where(df_no_empty_cells_label.applymap(lambda x: x == ''))[1]) == 0 ) 


def test_drop_nan_for_text():
    '''
    Test function to test the behaviour of drop_empty_rows function.
    The initial dataset can be composed of empty cells, or cells with
    no text inside (only ''). This kind of data are not so meaningful,
    so drop_empty_rows cares to remove these rows.
    The final dataset contains only rows with text.

    In this test function it is tested that there are no None cells 
    for the 'text' column, starting from a dataframe with None cells in 
    the 'text' column.

    Given:
    ========
    init_dataframe: pd.DataFrame
                    Original dataframe with None cells in 'text' column.

    Tests:
    ======
            The output dataframe should not contain any None cell for the
            'text' column.
    '''
    init_dataframe = pd.DataFrame({'text': ['a', None, 'c'], 'label':['0','0','1']})
    df_no_none_cells_text = drop_empty_rows(init_dataframe)
    # no None cell for text:
    assert( len(np.where(pd.isnull(df_no_none_cells_text))[0]) == 0 ) 

def test_drop_nan_for_label():
    '''
    Test function to test the behaviour of drop_empty_rows function.
    The initial dataset can be composed of empty cells, or cells with
    no text inside (only ''). This kind of data are not so meaningful,
    so drop_empty_rows cares to remove these rows.
    The final dataset contains only rows with text.

    In this test function it is tested that there are no None cells 
    for the 'label' column, starting from a dataframe with None cells in 
    the 'label' column.

    Given:
    ========
    init_dataframe: pd.DataFrame
                    Original dataframe with None cells in 'label' column.

    Tests:
    ======
            The output dataframe should not contain any None cell for the
            'label' column.
    '''
    init_dataframe = pd.DataFrame({'text': ['a', 'b', 'c'], 'label':['0',None,'1']})
    df_no_none_cells_label = drop_empty_rows(init_dataframe)
    # no None cell for label:
    assert( len(np.where(pd.isnull(df_no_none_cells_label))[1]) == 0 ) 
