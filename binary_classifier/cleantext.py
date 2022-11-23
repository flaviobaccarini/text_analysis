import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
from collections import defaultdict


# np.where(df.applymap(lambda x: x == '')) per vedere dove ho empty string


def drop_empty_rows(df):
    df_no_empty_rows = df.copy()
    
    #for col in columns:
    #    df_no_empty_rows = df_no_empty_rows[df_no_empty_rows[col].str.strip().astype(bool)]
    filter = df != ""
    df_no_empty_rows = df[filter]
    #indices_to_remove = df_no_empty_rows[df_no_empty_rows['text'] == ''].index
    #df_no_empty_rows.drop(index = indices_to_remove, inplace = True)
    df_no_empty_rows.dropna(axis = 0, how = 'any', inplace = True)

    return df_no_empty_rows

def rename_columns(df, columns):
    df_new_column_names = df.copy()
    df_new_column_names = df.loc[:, list(columns)] # COLUMN NUMBER 0: TEXT, COLUMN NUMBER 1: LABEL
    df_new_column_names.rename(columns = {columns[0]: 'text', columns[1]: 'label'}, inplace=True)
    #df_new_column_names.columns = ['text', 'label']
    return df_new_column_names


#convert to lowercase, strip and remove punctuations
def lower_strip(text):
    text_cleaned = text.lower() # lowercase
    text_cleaned = text_cleaned.strip()  # strip (elimina gli spazi prima e dopo)
    text_cleaned = re.sub(r'\s+', ' ', text_cleaned)   # elimina i white spaces 
    return text_cleaned

#TODO: QUESTA FUNZIONE DI RIMOZIONE DI EMOJI FORSE È DI TROPPO.. ELIMINANDO TUTTO CIÒ CHE NON È ALFANUMERICOD DOVREBBE BASTARE
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


def remove_urls_tags(text):
    text_cleaned = re.sub(r'http\S+', '', text) # remove url
    text_cleaned = re.compile('<.*?>').sub('', text_cleaned) # remove all chars between < >
    return text_cleaned


def remove_noalphanum(text):
    #will match anything that's not alphanumeric (also underscore!)
    text_cleaned = re.sub(r'[^a-zA-Z0-9]', ' ', text)

    # TODO: DECIDERE SE TENERE O NO
    # perchè le stopword come I, s, e simili esistono già e vengono eliminate da stopword
    # will match single characters: for example the s letters after the apostrophes
#    text_cleaned = re.sub(r' \w{1} |^\w{1} | $\w{1}', ' ', text_cleaned)
    return text_cleaned

def clean_text(text):
    text_cleaned = remove_urls_tags(text)
    text_cleaned = remove_emojis(text_cleaned)
    #text_cleaned = re.sub(r'\d+st|\d+nd|\d+rd|\d+th', '', text_cleaned) # eliminate: 1st, 2nd, 3rd and so on regarding the date
    text_cleaned = remove_noalphanum(text_cleaned)
    text_cleaned = lower_strip(text_cleaned) # lowercase and remove the whitespaces
    return text_cleaned

# STOPWORD REMOVAL
def stopword(text):
    words = [word for word in text.split() if word not in stopwords.words('english')]
    return ' '.join(words)

# This is a helper function to map NTLK position tags
def get_wordnet_pos(tag):
    # DEFAULT: NOUN
    # J : ADJECTIVE
    # V : VERB
    # R : ADVERB
    tags_dict = defaultdict(lambda: wordnet.NOUN, 
        {'J': wordnet.ADJ, 'V': wordnet.VERB,
         'R': wordnet.ADV})
    return tags_dict[tag[0]]

# Tokenize the sentence
def lemmatizer(text):
    wl = WordNetLemmatizer()
    word_pos_tags = nltk.pos_tag(word_tokenize(text)) # Get position tags
    lemma_words = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(lemma_words)

def finalpreprocess(text):
    return lemmatizer(stopword(clean_text(text)))