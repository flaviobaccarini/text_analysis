from read_write_data import read_data, write_data
import re, string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
#nltk.download('omw-1.4')
#nltk.download('wordnet')
#nltk.download('stopwords')
# Tools for vectorizing input data
from gensim.models import Word2Vec
import sys
import configparser


#for model-building
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import MinMaxScaler #fixed import

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
    #text_cleaned = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text_cleaned)  # remove punctuations
    text_cleaned = re.sub(r'\d+st|\d+nd|\d+rd|\d+th', ' ', text_cleaned) # eliminate: 1st, 2nd, 3rd and so on regarding the date
    #text_cleaned = re.sub(r'\d',' ',text_cleaned) #remove the numbers
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
    words = [wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(words)

def finalpreprocess(tweet):
    return lemmatizer(stopword(clean_tweet, tweet))


#building Word2Vec model
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec, tf_bool):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
        self.tf_bool = tf_bool
    def fit(self, X, y):
            return self
            
    def transform(self, X, y):
            new_X = np.array([
                np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                        or [np.zeros(self.dim)], axis=0)
                for words in X
            ])
            new_y = np.array(y.map({'real': 1, 'fake': 0}).astype(int))

            if self.tf_bool is True:
                new_X = tf.convert_to_tensor(new_X)
                new_y = tf.convert_to_tensor(new_y)

            return new_X, new_y

def clean_dataframes_write_csv(input_folder, output_folder):
    df_train, df_val, df_test = read_data(input_folder)
    df_train['clean_tweet'] = df_train['tweet']
    df_val['clean_tweet'] = df_val['tweet']
    df_test['clean_tweet'] = df_test['tweet']

    cleaned_tweet = [finalpreprocess(tweet_to_clean) for tweet_to_clean in tqdm(df_train['clean_tweet'])]
    df_train['clean_tweet'] = cleaned_tweet

    cleaned_tweet = [finalpreprocess(tweet_to_clean) for tweet_to_clean in tqdm(df_val['clean_tweet'])]
    df_val['clean_tweet'] = cleaned_tweet

    cleaned_tweet = [finalpreprocess(tweet_to_clean) for tweet_to_clean in tqdm(df_test['clean_tweet'])]
    df_test['clean_tweet'] = cleaned_tweet

    print(df_train['clean_tweet'].head())
    print("="*40)
    print(df_val['clean_tweet'].head())

    write_data((df_train, df_val, df_test), output_folder = output_folder)


def vectorize_data(data, vocabulary, tf_bool):

    X_tok= [nltk.word_tokenize(i) for i in data['clean_tweet']]  
 
    model = Word2Vec(vocabulary,min_count=1)   
    w2v = dict(zip(model.wv.index_to_key, model.wv.vectors)) 

    modelw = MeanEmbeddingVectorizer(w2v, tf_bool)
    # converting text to numerical data using Word2Vec
    X_vectors_w2v, y_vector = modelw.transform(X_tok, data['label'])


    return X_vectors_w2v, y_vector

def get_vocabulary(df_train, df_val):
    df_train_val = pd.concat([df_train, df_val], ignore_index = True)
    vocabulary = [nltk.word_tokenize(i) for i in df_train_val['clean_tweet']]
    return vocabulary


def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    input_folder = config_parse.get('INPUT_OUTPUT', 'input_folder')
    output_folder = config_parse.get('INPUT_OUTPUT', 'output_folder')

    clean_dataframes_write_csv(input_folder, output_folder)

if __name__ == '__main__':
    main()

'''
#clean_dataframes_write_csv()
df_train, df_val, df_test = read_data(name_folder = 'datasets_modified')
#Word2Vec
# Word2Vec runs on tokenized sentences

X_train_tok= [nltk.word_tokenize(i) for i in df_train['clean_tweet']]  
X_val_tok= [nltk.word_tokenize(i) for i in df_val['clean_tweet']]

df_train_val = pd.concat([df_train, df_val], ignore_index = True)
df_train_val['clean_text_tok']=[nltk.word_tokenize(i) for i in df_train_val['clean_tweet']]
model = Word2Vec(df_train_val['clean_text_tok'],min_count=1)     
w2v = dict(zip(model.wv.index_to_key, model.wv.vectors)) 

modelw = MeanEmbeddingVectorizer(w2v)
# converting text to numerical data using Word2Vec
X_train_vectors_w2v = modelw.transform(X_train_tok)
X_val_vectors_w2v = modelw.transform(X_val_tok)

y_train = df_train['label'].map({'real': 1, 'fake': 0}).astype(int)
y_test = df_val['label'].map({'real': 1, 'fake': 0}).astype(int)


#DA QUI FINISCE LA PARTE DI PREPROCCESING
#TODO FARE UN MAIN IN CUI AL SUO INTERNO INSERISCI TUTTO IL PREPROCESSING

lr_w2v=LogisticRegression(solver = 'liblinear', C=10, penalty = 'l2')
lr_w2v.fit(X_train_vectors_w2v, y_train)  #model

#Predict y value for test dataset
y_predict = lr_w2v.predict(X_val_vectors_w2v)
y_prob = lr_w2v.predict_proba(X_val_vectors_w2v)[:,1]

print(classification_report(y_test,y_predict))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))
 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)

#FITTING THE CLASSIFICATION MODEL using Naive Bayes(tf-idf)
nb = MultinomialNB()

scaler = MinMaxScaler((0, np.max(X_train_vectors_w2v)))
X_train = scaler.fit_transform(X_train_vectors_w2v)
X_test = scaler.transform(X_val_vectors_w2v)

nb.fit(X_train, y_train)  
#Predict y value for test dataset
y_predict = nb.predict(X_test)
y_prob = nb.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_predict))
print('Confusion Matrix:',confusion_matrix(y_test, y_predict))
 
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print('AUC:', roc_auc)
'''