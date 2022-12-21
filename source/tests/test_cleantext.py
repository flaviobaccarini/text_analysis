'''
Test functions for testing the functions inside the cleantext module.
'''
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
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

def test_lower_no_capital_letters():
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
         whitespace in the middle (replaced by only a single whitespace: ' ').
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
         the end: it is replaced by a whitespace.
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
         the beninning: it is replaced by a whitespace.
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
                   Text with URL at the middle of the text.
    
    Tests:
    ======
         The final text after the function doesn't include the URL within
         the text, but it's replaced by a whitespace.
    '''
    text_with_url = 'The wikipedia site https://en.wikipedia.org/wiki is here'
    text_without_url =  remove_urls_tags(text_with_url)
    assert(text_without_url == 'The wikipedia site  is here')


def test_rm_no_tags_or_url():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In this test function the input text doesn't contain any tags or URLs,
    so it should not change after the function,

    Given:
    ======
    text_no_tag_or_url: str
                        Text with no tags or URLs. 
    
    Tests:
    ======
         The final output text should be equal to the input text, since
         it doesn't contain any URLs or tags.
    '''
    text_no_tag_or_url = 'Hello World'
    text_without_url =  remove_urls_tags(text_no_tag_or_url)
    assert(text_without_url == 'Hello World')

def test_rm_only_url():
    '''
    This test function tests the behaviour of remove_url_tag function.
    In this test function the input text contains only an URL.

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
            tag at the beginning: it is replaced by a whitespace.
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
            tag at the end: it is replaced by a whitespace.
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
            two tags (so the text between the two tags).
    '''
    text_with_tags = '<TEXT> Hello World <TEXT>'
    text_without_tag = remove_urls_tags(text_with_tags)
    assert(text_without_tag == ' Hello World ')

def test_rm_punctuations():
    '''
    This test function tests the behaviour of remove_noalphanum function.
    In particular, this test function tests that all the punctuation is eliminated 
    from the initial string.

    Given:
    ======
    text_with_punctuations: str
                            Text string with punctuation.

    Tests:
    ======
          The output string should not contain any punctuation.
          Punctuation is replaced by a whitespace.
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
          The output string should not contain any emoticons (each emoticon 
          is replaced by a whitespace).
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
    text_with_quot_marks: str
                          Text string with quotation marks.

    Tests:
    ======
          The output string should not contain any quotation marks
          (replaced with whitespaces).
    '''
    text_with_quot_marks = 'try with "hello"'
    text_no_quot_marks = remove_noalphanum(text_with_quot_marks)
    assert(text_no_quot_marks == "try with  hello ")

def test_rm_apostrophe():
    '''
    This test function tests the behaviour of remove_noalphanum function.
    In particular, this test function tests that apostrophes are eliminated 
    from the input text.

    Given:
    ======
    text_with_apostrophe: str
                          Text string with apostrophes.

    Tests:
    ======
          The output string should not contain any apostrophe.
    '''
    text_with_aposrophe = "i'm flavio let's try"
    text_no_apostrophe = remove_noalphanum(text_with_aposrophe)
    assert(text_no_apostrophe == "i m flavio let s try")

def test_clean_text_capital_letters_punctuation():
    '''
    Test function for clean_text.
    In this test function it's tested if clean_text correctly removes
    punctuation and make all the letters lowercase.

    Given:
    ======
    text_punctuation_capital_letters: str
                                      Text with capital letters and punctuation.
    
    Tests:
    ======
            if the output string doesn't contain punctuation and if it has all the
            letters lowercase.
    '''
    text_punctuation_capital_letters = 'Hello World, i am Flavio!'
    text_no_punct_lowercase = clean_text(text_punctuation_capital_letters)
    assert(text_no_punct_lowercase == 'hello world i am flavio')

def test_clean_text_capital_letters_tag():
    '''
    Test function for clean_text.
    In this test function it's tested if clean_text correctly removes
    tags and make all the letters lowercase.

    Given:
    ======
    text_tags_capital_letters: str
                               Text with capital letters and tags.
    
    Tests:
    ======
            if the output string doesn't contain tags and if it has all the
            letters lowercase.
    '''
    text_tags_capital_letters = '<TEXT> HELLO WORLD <TEXT>'
    text_no_punct_lowercase = clean_text(text_tags_capital_letters)
    assert(text_no_punct_lowercase == 'hello world')

def test_clean_text_capital_letters_url():
    '''
    Test function for clean_text.
    In this test function it's tested if clean_text correctly removes
    URL and make all the letters lowercase.

    Given:
    ======
    text_url_capital_letters: str
                              Text with capital letters and a URL.
    
    Tests:
    ======
            if the output string doesn't contain the URL and if it has all the
            letters lowercase.
    '''
    text_url_capital_letters = 'Visit the Wikipedia site http://wikipedia.com'
    text_no_punct_lowercase = clean_text(text_url_capital_letters)
    assert(text_no_punct_lowercase == 'visit the wikipedia site')

def test_clean_text_whitespaces_punctuation():
    '''
    Test function for clean_text.
    In this test function it's tested if clean_text correctly removes
    big whitespaces within the text and also the punctuation.

    Given:
    ======
    text_big_whitespace_punctuation: str
                                     Text with a big whitespace within the text
                                     and some punctuation.
    
    Tests:
    ======
            if the output string doesn't contain the big whitespace (but only a single
            character whitespace) and if it doesn't contain punctuation.
    '''
    text_url_capital_letters = 'hello world,      i     am   flavio!'
    text_no_punct_lowercase = clean_text(text_url_capital_letters)
    assert(text_no_punct_lowercase == 'hello world i am flavio')

def test_clean_text_specialchars_url():
    '''
    Test function for clean_text.
    In this test function it's tested if clean_text correctly removes
    special characters and URL from the input text.

    Given:
    ======
    text_URL_specialchars_punctuation: str
                                       Text with a URL and special characters.
    
    Tests:
    ======
            if the output string doesn't contain the URL and special characters.
    '''
    text_url_capital_letters = 'visit the #wikipedia site http://wikipedia.com'
    text_no_punct_lowercase = clean_text(text_url_capital_letters)
    assert(text_no_punct_lowercase == 'visit the wikipedia site')

def test_clean_text_specialchars_cap_letters_emoticons():
    '''
    Test function for clean_text.
    In this test function it's tested if clean_text correctly removes
    special characters, emoticons and makes all the capital letters
    as lowercase

    Given:
    ======
    text_specialchars_capletters_emoticons: str
                                            Text with special characters, capital letters
                                            and emoticons.
    
    Tests:
    ======
            if the output string doesn't contain the special characters, emoticons and 
            all the letters are lowercase.
    '''
    text_url_capital_letters = 'Random emoticon \U00002666 only for #Testing purpose'
    text_no_punct_lowercase = clean_text(text_url_capital_letters)
    assert(text_no_punct_lowercase == 'random emoticon only for testing purpose')


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

def test_stopword_capital_letters():
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
    input text there are stop words with capital letters.

    Given:
    ======
    text_with_capital_stopwords: str
                                 Text with stop words with capital letters.
    
    Tests:
    ======
            The output string should be equal to the input text, since
            it the stop words present in nltk are all lowercase and so
            it is not able to remove stop words with capital letters.
    '''
    text_with_stopwords = "A random sentence To see How It works"
    text_no_stopword = stopword(text_with_stopwords)
    assert(text_no_stopword == 'A random sentence To see How It works')

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
    In this test function it's tested what is a result for the lemmatization of some 
    words.

    Given:
    ======
    text_to_lemmatize: str
                       Text string that needs to be lemmatized.
    
    Tests:
    ======
            The output text should be the lemmatized text.
    '''
    text_to_lemmatize = "i am playing cards"
    lemmatized_text = lemmatizer(text_to_lemmatize)
    assert(lemmatized_text == 'i be play card')

def test_lemmatizer_capital_text():
    '''
    Test function to test the behaviour of lemmatizer function.
    The lemmatizer function takes a text (string) as input and lemmatize the text.
    This test function tests what's the result if the input text contains some 
    words that need to be lemmatized, but with capital letters.

    Given:
    ======
    text_to_lemmatize_with_capital: str
                                    Text with words that need to be lemmatized with
                                    capital letters.
    
    Tests:
    ======
            The output text should be equal to the input text string
            since the lemmatization works only with lowercase words.
    '''
    text_to_lemmatize_with_capital = "I Am Playing Cards"
    lemmatized_text = lemmatizer(text_to_lemmatize_with_capital)
    assert(lemmatized_text == "I Am Playing Cards")

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
    In this test function it's tested if the get_wordnet_pos well
    recognize the words within a sentence.

    Given:
    ======
    text_for_tag: str
                  Text from which to extrapolate wordnet tags.
    
    Tests:
    ======
            if the get_wordnet_pos can well recognize if a word
            is a noun, verb, adverb or adjective in a sentence.
    '''
    text_for_tag = "it is incredibly beautiful"
    tags = nltk.pos_tag(word_tokenize(text_for_tag))
    wordnet_pos = [get_wordnet_pos(tag[1]) for tag in tags]
    assert(wordnet_pos == ['n', 'v', 'r', 'a']) # NOUN, VERB, ADVERB, ADJECTIVE

def test_finalpreprocess_capletters_stopword_lemmatization():
    '''
    Test function for finalpreprocess.
    In this test function it's tested if the function correctly
    replace all the capital letters with lowercase letters, remove
    the punctuation, stop words and lemmatize the text.

    Given:
    =======
    sentence: str
              Single sentence with capital letters, punctuation,
              stop words and not lemmatized words.

    Tests:
    ======
            if the finalpreprocess function correctly replace all the 
            capital letters with lowercase letters, remove the punctuation,
            stop words and lemmatize all the words. 
    '''
    sentence = 'I Am Playing Basketball'
    preprocessed_sentence = finalpreprocess(sentence)
    assert(preprocessed_sentence == 'play basketball')

def test_finalpreprocess_punctuation_specialchars_stopword_lemmatization():
    '''
    Test function for finalpreprocess.
    In this test function it's tested if the function correctly
    remove the punctuation, stop words, special characters
    and lemmatize the text.

    Given:
    =======
    sentence: str
              Single sentence with punctuation, special characters,
              stop words and not lemmatized words.

    Tests:
    ======
            if the finalpreprocess function correctly remove the punctuation,
            special characters, stop words and lemmatize all the words. 
    '''
    sentence = 'hi! i am playing #basketball!'
    preprocessed_sentence = finalpreprocess(sentence)
    assert(preprocessed_sentence == 'hi play basketball')

def test_finalpreprocess_url_stopword():
    '''
    Test function for finalpreprocess.
    In this test function it's tested if the function correctly
    remove the URL and the stop words.

    Given:
    =======
    sentence: str
              Single sentence with an URL and stop words.

    Tests:
    ======
            if the finalpreprocess function correctly remove the URL,
            and stop words. 
    '''
    sentence = 'please take a look at this website http://wikipedia.com'
    preprocessed_sentence = finalpreprocess(sentence)
    assert(preprocessed_sentence == 'please take look website')

def test_rename_columns():
    '''
    Test function to test the behaviour of rename_columns function.
    In this test function it's tested if the rename_columns function correctly
    rename the columns' names.

    Given:
    =======
    dataframe_init_columns: pd.DataFrame
                            Oriiginal dataframe with initial column names.

    Tests:
    ======
            This test function tests that in the output dataframe there are
            the columns named: 'text', 'label'.
    '''
    dataframe_init_column = pd.DataFrame(columns=('original_text', 'original_target'))
    df_correct_col_names = rename_columns(dataframe_init_column,
                                          text_column_name = 'original_text', 
                                          label_column_name = 'original_target')
    assert(df_correct_col_names.columns.values.tolist() == ['text', 'label']) 


def test_rename_columns_number_of_columns():
    '''
    Test function to test the behaviour of rename_columns function.
    In this test function it's tested if the rename_columns function correctly
    rename the columns' names in the case where the original dataframe contains
    more than two columns.

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
    assert(df_correct_col_names.columns.values.tolist() == ['text', 'label']) 
    
def test_rename_columns_inverted_order():
    '''
    Test function to test the behaviour of rename_columns function.
    In this test function it is tested that the output dataframe contains two columns
    ('text' and 'label') also if in the original dataframe there is first the label 
    column than the text column.

    Given:
    =======
    dataframe_init_columns: pd.DataFrame
                            Oriiginal dataframe with initial column names.

    Tests:
    ======
            if the function doesn't depend on the order of the columns in the
            original dataframe: it means that the output dataframe contains 
            always two columns with name 'text' and 'label' that correctly corresponds
            to the original text and label columns.
    '''  
    dataframe_init_column = pd.DataFrame({'original_target': ['label'], 'original_text': ['text']})
    df_correct_col_names = rename_columns(dataframe_init_column,
                                          text_column_name = 'original_text', 
                                          label_column_name = 'original_target')
    assert(df_correct_col_names['text'][0] == 'text')
    assert(df_correct_col_names['label'][0] == 'label') 


def test_drop_empty_for_text():
    '''
    Test function to test the behaviour of drop_empty_rows function.
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


def test_drop_none_for_text():
    '''
    Test function to test the behaviour of drop_empty_rows function.
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

def test_drop_none_for_label():
    '''
    Test function to test the behaviour of drop_empty_rows function.
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
