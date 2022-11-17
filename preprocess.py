from binary_classifier.read_write_data import read_data, write_data, split_dataframe, clean_dataframe
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from nltk.stem import SnowballStemmer # for stemming; but now used lemmatization
from tqdm import tqdm
from pathlib import Path
#nltk.download('omw-1.4')
#nltk.download('wordnet')
#nltk.download('stopwords')
# Tools for vectorizing input data
import sys
import configparser
import pandas as pd


#TODO: sposta questa funzione in preprocess e aggiungici una parte per eliminare eventuali righe vuote
def clean_dataframe(dfs_raw, column_names):
    df_raw_train, df_raw_val, df_raw_test = dfs_raw
    correct_dataframes = []

    for dataframe in (df_raw_train, df_raw_val, df_raw_test):
        df_new_correct = dataframe.loc[:, list(column_names)] # COLUMN NUMBER 0: TEXT, COLUMN NUMBER 1: LABEL
        dict_new_names = {column_names[0]: 'text', column_names[1]: 'label'}
        df_new_correct = df_new_correct.rename(dict_new_names, axis = 'columns')
        correct_dataframes.append(df_new_correct)
    
    return correct_dataframes

#convert to lowercase, strip and remove punctuations
def lower_strip(text):
    text_cleaned = text.lower() # lowercase
    text_cleaned = text_cleaned.strip()  # strip (elimina gli spazi prima e dopo)
    text_cleaned = re.sub('\s+', ' ', text_cleaned)   # elimina i white spaces lasciati dalla punteggiatura
    return text_cleaned

def remove_numbers(text):
    text_cleaned = re.sub(r'\d+st|\d+nd|\d+rd|\d+th', ' ', text) # eliminate: 1st, 2nd, 3rd and so on regarding the date
    text_cleaned = re.sub(r'\d+k', ' ', text_cleaned) # remove something like 1k or 10k etc..
    text_cleaned = re.sub(r'\d+', ' ', text_cleaned) # remove all the numbers
    return text_cleanedpyth

def clean_tweet(text):
    text_cleaned = re.sub(r'http\S+', '', text) # remove url
    text_cleaned = re.compile('<.*?>').sub('', text_cleaned) # remove all chars between < >
    text_cleaned = remove_emojis(text_cleaned)
    #text_cleaned = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text_cleaned)  # remove punctuations
    text_cleaned = re.sub(r'\d+st|\d+nd|\d+rd|\d+th', ' ', text_cleaned) # eliminate: 1st, 2nd, 3rd and so on regarding the date
    #text_cleaned = re.sub(r'\d',' ',text_cleaned) #remove the numbers
    #text_cleaned = remove_numbers(text_cleaned)
    
    text_cleaned = re.sub(r'[^\w]', ' ', text_cleaned) # will match anything that's not alphanumeric or underscore
    text_cleaned = text_cleaned.replace('"', ' ') 
    text_cleaned = text_cleaned.replace("'", ' ') 
    text_cleaned = lower_strip(text_cleaned) # lowercase and remove the whitespaces
    return text_cleaned

# STOPWORD REMOVAL
def stopword(clean_tweet, text):
    text_cleaned = clean_tweet(text)
    words = [word for word in text_cleaned.split() if word not in stopwords.words('english')]
    return ' '.join(words)

def remove_emojis(text):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', text)

def get_lemmatizer():
    #LEMMATIZATION
    # Initialize the lemmatizer
    wl = WordNetLemmatizer()
    return wl

# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Tokenize the sentence
def lemmatizer(text):
    wl = get_lemmatizer()
    word_pos_tags = nltk.pos_tag(word_tokenize(text)) # Get position tags
    lemma_words = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(lemma_words)

def finalpreprocess(tweet):
    return lemmatizer(stopword(clean_tweet, tweet))


def clean_dataframes_write_csv(dfs_cleaned, output_folder, analysis_name):
    df_train, df_val, df_test = dfs_cleaned

    for dataframe in (df_train, df_val, df_test):
        dataframe['clean_text'] = dataframe['text']
        cleaned_text = [finalpreprocess(text_to_clean) for text_to_clean in tqdm(dataframe['clean_text'])]
        indices_to_remove = [index for index, text in enumerate(cleaned_text) if text == '']
        dataframe['clean_text'] = cleaned_text
        dataframe.drop(indices_to_remove, axis = 0, inplace = True)
        dataframe.reset_index(inplace = True)
        dataframe.drop('index', axis = 'columns')

    write_data((df_train, df_val, df_test), output_folder = output_folder, analysis = analysis_name)

def print_cleaned_data(dfs_cleaned):
    df_train, df_valid, df_test = dfs_cleaned
    print(df_train['clean_text'].head())
    print("="*40)
    print(df_valid['clean_text'].head())
    print("="*40)

    print("Some random texts:\n" + "="*40)
    for index, row in df_train.sample(n = 3).iterrows():
        print("\nOriginal text:\n" + "="*40) 
        print(row['text'])
        print("\nCleaned text:\n" + "="*40)
        print(row['clean_text'])

def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    #input_folder = config_parse.get('INPUT_OUTPUT', 'input_folder')
    #output_folder = config_parse.get('INPUT_OUTPUT', 'folder_preprocessed_datasets')

    analysis_name = config_parse.get('INPUT_OUTPUT', 'analysis')
    seed = int(config_parse.get('PREPROCESS', 'seed'))
    dataset_folder = Path('datasets') / analysis_name

    dfs_raw = read_data(dataset_folder)
    if len(dfs_raw) != 3:

        fractions =    (float(config_parse.get('PREPROCESS', 'train_fraction')),
                        float(config_parse.get('PREPROCESS', 'test_fraction')))
        dfs_raw = split_dataframe(dfs_raw, fractions, seed)

    column_names = (config_parse.get('PREPROCESS', 'column_name_text'), 
                    config_parse.get('PREPROCESS', 'column_name_label'))

    dfs_cleaned = clean_dataframe(dfs_raw, column_names)

    output_folder = Path('preprocessed_datasets') / analysis_name
    output_folder.mkdir(parents=True, exist_ok=True)

    clean_dataframes_write_csv(dfs_cleaned, output_folder, analysis_name)

    print_cleaned_data(dfs_cleaned)

if __name__ == '__main__':
    main()
