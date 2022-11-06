from read_write_data import read_data, write_data
import re, string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from tqdm import tqdm
import pandas as pd
import numpy as np
#nltk.download('omw-1.4')
#nltk.download('wordnet')
#nltk.download('stopwords')
# Tools for creating ngrams and vectorizing input data
from gensim.models import Word2Vec


#convert to lowercase, strip and remove punctuations
def lower_strip(text):
    text_cleaned = text.lower() # lowercase
    text_cleaned = text_cleaned.strip()  # strip (elimina gli spazi prima e dopo)
    text_cleaned = re.sub('\s+', ' ', text_cleaned)   # elimina i white spaces lasciati dalla punteggiatura
    return text_cleaned

def clean_tweet(text):
    text_cleaned = re.sub(r'http\S+', '', text) # remove url
    text_cleaned = re.compile('<.*?>').sub('', text_cleaned) # remove all chars between < >
    text_cleaned = remove_emojis(text_cleaned)
    text_cleaned = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text_cleaned)  # remove punctuations
    text_cleaned = re.sub(r'\d+st|\d+nd|\d+rd|\d+th', ' ', text_cleaned) # eliminate: 1st, 2nd, 3rd and so on regarding the date
    text_cleaned = re.sub(r'\d',' ',text_cleaned) #remove the numbers
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
    words = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(words)

def finalpreprocess(tweet):
    return lemmatizer(stopword(clean_tweet, tweet))


#building Word2Vec model
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
def fit(self, X, y):
        return self
def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

def clean_dataframes_write_csv():
    df_train, df_val, df_test = read_data()
    df_train['clean_tweet'] = df_train['tweet']
    df_val['clean_tweet'] = df_val['tweet']
    cleaned_tweet = [finalpreprocess(tweet_to_clean) for tweet_to_clean in tqdm(df_train['clean_tweet'])]
    df_train['clean_tweet'] = cleaned_tweet

    cleaned_tweet = [finalpreprocess(tweet_to_clean) for tweet_to_clean in tqdm(df_val['clean_tweet'])]
    df_val['clean_tweet'] = cleaned_tweet
    print(df_train['clean_tweet'].head())
    print("="*40)
    print(df_val['clean_tweet'].head())

    write_data((df_train, df_val, df_test), name_folder = 'datasets_modified')

#clean_dataframes_write_csv()
df_train, df_val, df_test = read_data(name_folder = 'datasets_modified')
#Word2Vec
# Word2Vec runs on tokenized sentences
'''
X_train_tok= [nltk.word_tokenize(i) for i in df_train['clean_tweet']]  
X_val_tok= [nltk.word_tokenize(i) for i in df_val['clean_tweet']]

df_train_val = pd.concat([df_train, df_val], ignore_index = True)
df_train_val['clean_text_tok']=[nltk.word_tokenize(i) for i in df_train_val['clean_tweet']]
model = Word2Vec(df_train_val['clean_text_tok'],min_count=1)     
w2v = dict(zip(model.wv.index_to_key, model.wv.syn0)) 

modelw = MeanEmbeddingVectorizer(w2v)
# converting text to numerical data using Word2Vec
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_val_vectors_w2v = modelw.transform(X_val_tok)

print(X_train_tok)
'''