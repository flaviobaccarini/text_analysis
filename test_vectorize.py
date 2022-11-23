from binary_classifier.vectorize_data import vectorize_X_data_lr, vectorize_X_data_tf
from binary_classifier.vectorize_data import calculate_max_len, get_vocabulary
from binary_classifier.vectorize_data import MeanEmbeddingVectorizer
from binary_classifier.vectorize_data import init_vector_layer, tocat_encode_labels
import numpy as np
import nltk
from nltk import word_tokenize

from hypothesis import strategies as st
from hypothesis import given

def test_vocabulary():
    all_words = ['hello world', 'what a beautiful day',
                'random string', 'another random text', 'a little bit of fantasy',
                'try with this','1 + 1 is equal to 2', '1 + 1 = 2']

    voc = get_vocabulary(all_words)
    assert(len(voc) == 31) # total nr. of words == 31
    assert(len(np.unique(voc)) == 24) # only 24 unique words

    voc_unique = get_vocabulary(all_words, unique = True)
    assert(len(voc_unique) == 24)

@given(list_words = st.lists(st.text(max_size = 10)))
def test_voc_rand_text(list_words):
    voc = get_vocabulary(list_words)
    nr_of_words = 0
    for item in list_words:
        nr_of_words += (len(nltk.word_tokenize(item)))
  
    assert(len(voc) == nr_of_words)

    voc_unique = get_vocabulary(list_words, unique = True)

    word_tokenize_uniq = [nltk.word_tokenize(word) for word in list_words]
    flat_list = [item for sublist in word_tokenize_uniq for item in sublist]
    uniq_words = " ".join(set(flat_list))
    assert(len(uniq_words.split()) ==  len(voc_unique))
