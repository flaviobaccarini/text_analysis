'''
CLEANING TEXT MODULE
====================
In this module there are functions for cleaning and 
preprocessing the text data.
'''
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from collections import defaultdict
import pandas as pd


def drop_empty_rows(df_to_clean: pd.DataFrame) -> pd.DataFrame:
    '''
    Function used for cleaning the dataframe from the cells containing 
    NaN value or empty cells (cells contain '').

    Parameters:
    ===========
    df_to_clean: pd.DataFrame
                 Dataframe that needs to be cleaned from empty cells.
    
    Returns:
    =========
    df_no_empty_rows: pd.DataFrame
                      The initial dataframe cleaned from empty cells.
    '''

    df_no_empty_rows = df_to_clean.copy()
    filter = df_no_empty_rows != ""
    df_no_empty_rows = df_no_empty_rows[filter]
    df_no_empty_rows.dropna(axis = 0, how = 'any', inplace = True)

    return df_no_empty_rows

def rename_columns(df: pd.DataFrame, 
                   text_column_name: str,
                   label_column_name: str) -> pd.DataFrame:
    '''
    Function used for renaming the dataframe columns.
    The inital dataframe can have multiple columns, while after
    this function only two columns are kept: "text" and "label" columns.

    Parameters:
    ===========
    df: pd.DataFrame
        Initial dataframe with all the columns.

    text_column_name: str
                      String that corresponds to the 
                      initial text column name
                      This column will be kept in the final dataframe,
                      changing its name in "text".

    label_column_name: str
                       String that corresponds to the 
                       initial label column name
                       This column will be kept in the final dataframe,
                       changing its name in "label".    
    
    Returns:
    =========
    df_new_column_names: pd.DataFrame
                         The dataframe with only the text and the label
                         columns.
    '''

    df_new_column_names = df.copy()
    df_new_column_names = df.loc[:, [text_column_name, label_column_name]]
    df_new_column_names.rename(columns = {text_column_name: 'text',
                                          label_column_name: 'label'}, inplace=True)
    return df_new_column_names


def lower_strip(text: str) -> str:
    '''
    This function makes all the letters inside the text lowercase,
    remove the whitespaces at the beginning and at the end of the text
    and replace the big whitespaces (so whitespaces with more than 1
    characters for example '   ') with a single whitespace (' ').

    Parameters:
    ===========
    text: str
          Text to convert all letters as lowercase and remove the 
          whitespaces.

    Returns:
    =========
    text_cleaned: str
                  Text with all lowercase letters, without whitespaces
                  at the beginning or at the end of the text and without
                  big whitespaces within the text.
    '''
    text_cleaned = text.lower() # lowercase
    text_cleaned = text_cleaned.strip()  # strip 
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned) # remove whitespaces
    return text_cleaned


def remove_urls_tags(text: str) -> str:
    '''
    This function removes from the text all the HTML/JSON tags
    and the URLs.

    Parameters:
    ===========
    text: str
          Text to remove HTML/JSON tags and URLs.

    Returns:
    =========
    text_cleaned: str
                  Text without HTML/JSON tags and URLs.
    '''
    text_cleaned = re.sub(r'http\S+', '', text) # remove url
    text_cleaned = re.compile('<.*?>').sub('', text_cleaned) # remove all chars between < >
    return text_cleaned


def remove_noalphanum(text: str) -> str:
    '''
    This function removes from the text everything but
    alphanumeric characters. In this way, all the punctuations,
    the emoticons, the special characaters (such as '#' '@' '-' etc...)
    can be easily removed.

    Parameters:
    ===========
    text: str
          Text to keep only alhpanumeric characters.

    Returns:
    =========
    text_cleaned: str
                  Text with only alphanumeric characters.
    '''
    #will match anything that's not alphanumeric (also underscore!)
    text_cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text_cleaned

def clean_text(text_to_clean: str) -> str:
    '''
    Function that cleans the text removing HTML/JSON tags, 
    URLs and everything that is not alphanumeric.
    Finally it converts all the letters to lowercase, 
    it removes all the whitespaces at beginning or at the end
    of the text and it replaces unnecessary multiple 
    whitespaces with one single whitespace.

    Parameters:
    ===========
    text_to_clean: str
                   Initial text to clean.

    Returns:
    =========
    text_cleaned: str
                  Text cleaned with only lowercase letters,
                  no tags or urls, no emoticons or non alphanumeric
                  characters, no useless whitespaces.
    '''
    text_cleaned = remove_urls_tags(text_to_clean)
    text_cleaned = remove_noalphanum(text_cleaned)
    text_cleaned = lower_strip(text_cleaned) # lowercase and remove the whitespaces
    return text_cleaned

def stopword(text_stopword: str) -> str:
    '''
    Function to remove stop words from the text. Stop words are words
    in a stop list, which are filtered out because they are insignificant.
    To see the full list of the words contained in the stop list follow these
    code lines:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')
    print(stopwords.words('english')) 

    Parameters:
    ============
    text_stopword: str
                   Text with the stop words.

    Returns:
    =========
    text_no_stopwords: str
                       Text without the stop words.
    '''
    words = [word for word in text_stopword.split() 
                        if word not in stopwords.words('english')]
    text_no_stopwords = ' '.join(words)
    return text_no_stopwords


def get_wordnet_pos(tag: str) -> str:
    '''
    Helper function to map NTLK position tags.
    If the tag starts with J, it means the word is an adjective,
    if it starts with V the word is a verb,
    if it starts with R the word is an adverb.
    In the all the other cases words are considered as nouns.

    Parameters:
    ===========
    tag: str
         NLTK position tag

    Returns:
    ========
         str
         String that corresponds to the type 
         of the word analyzed (noun, adjective, adverb, verb).
    '''
    # DEFAULT: NOUN
    # J : ADJECTIVE
    # V : VERB
    # R : ADVERB
    tags_dict = defaultdict(lambda: wordnet.NOUN, 
        {'J': wordnet.ADJ, 'V': wordnet.VERB,
         'R': wordnet.ADV})
    return tags_dict[tag[0]]

def lemmatizer(text: str) -> str:
    '''
    Function for the lemmatization of the text.

    Parameters:
    ===========
    text: str
          Text that has to be lemmatized.

    Returns:
    ========
    lemma_text: str
                The text after lemmatization process.
    '''
    wl = WordNetLemmatizer()
    word_pos_tags = nltk.pos_tag(word_tokenize(text)) # Get position tags
    # Map the position tag and lemmatize the word/token
    lemma_words = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) 
                        for idx, tag in enumerate(word_pos_tags)] 
    lemma_text = " ".join(lemma_words)
    return lemma_text

def finalpreprocess(text):
    '''
    Function that makes all the preprocess of the text.
    First it applies the clean_text function, which removes
    URLs, tags, punctuations, emoticons, convert all the 
    letters to lowercase and remove unnecessary whitespaces.
    Then it applies the stopword function that removes the stop 
    words inside the text.
    Finally the application of lemmatizer function for the 
    lemmatization of the text.

    Parameters:
    text: str
          Initial text to be preprocessed.

    Returns:
          str
          Final text completely preprocessed
          (lemmatized, no stop words, no URLS, no tags,
          no punctuations, all letters lowercase,
          no unnecessary whitespaces). 
    '''
    return lemmatizer(stopword(clean_text(text)))
