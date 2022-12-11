'''
Test functions for testing the functions inside the vectorize_data module.
'''
from text_analysis.vectorize_data import vectorize_X_data_lr, vectorize_X_data_tf
from text_analysis.vectorize_data import calculate_max_len, get_vocabulary
from text_analysis.vectorize_data import MeanEmbeddingVectorizer
from text_analysis.vectorize_data import init_vector_layer, tocat_encode_labels
from text_analysis.vectorize_data import flatten_unique_voc
import numpy as np
import nltk
from hypothesis import strategies as st
from hypothesis import given
import string
import random
from gensim.models import Word2Vec
import tensorflow as tf

def test_vocabulary():
    '''
    Test function to test the behaviour of get_vocabulary function.
    The input to get_vocabulary should be a list of strings containing all 
    the sentences from which to extrapolate the vocabulary.
    The output is a list of lists containg all the single words.
    '''
    # some random strings
    all_words = ['hello world', 'what a beautiful day',
                'random string', 'another random text', 'a little bit of fantasy',
                'try with this','1 + 1 is equal to 2', '1 + 1 = 2']

    # get vocabulary
    sentences = get_vocabulary(all_words)

    # total nr. of sentences == 8:
    assert(len(sentences) == 8) 

    # now count the number of single words:
    nr_of_words = 0
    for sentence in sentences:
        nr_of_words += len(sentence)
    # total nr of words == 31    
    assert(nr_of_words == 31) 

@given(list_words = st.lists(st.text(max_size = 10)))
def test_voc_rand_text(list_words):
    '''
    Test function to test the behaviour of get_vocabulary function
    with some random text data generated by a strategy.
    The input to get_vocabulary should be a list of sentences containing all 
    the vocabulary.
    The output is a list containg sub-lists with all single words from the sentences.

    @given:
    list_words: list[str]
                Sequence of sentences to get the vocabulary.
    '''
    # list of sentences
    sentences = get_vocabulary(list_words)

    # nr of list inside sentnces is equal to the number of sentences
    assert(len(sentences) == len(list_words))

    # count how many words for each sentence
    nr_of_words = sum([len(nltk.word_tokenize(sentence)) for sentence in list_words])
    nr_of_words_from_func = sum([len(sentence) for sentence in sentences])

    assert(nr_of_words == nr_of_words_from_func) 


def test_flatten_vocabulary():
    '''
    Test function to test the behaviour of flatten_unique_voc function.
    The input to flatten_unique_voc should be a list of sub-lists
    of strings containing all the sentences tokenized.
    The output is a single list of strings containing 
    all the unique vocabulary tokenized words.
    '''
    # all sentences
    all_words = ['hello world', 'what a beautiful day',
                'random string', 'another random text', 'a little bit of fantasy',
                'try with this','1 + 1 is equal to 2', '1 + 1 = 2']

    # get the vocabulary: sentences is a list of lists of strings
    sentences = get_vocabulary(all_words)

    # flatten the vocabulary
    voc_unique = flatten_unique_voc(sentences)
    assert(len(voc_unique) == 24) # only 24 unique words in the sentences

@given(list_words = st.lists(st.text(max_size = 10)))
def test_flatten_rand_text(list_words):
    '''
    Test function to test the behaviour of flatten_unique_voc function
    with some random text data generated by a strategy.
    The input to flatten_unique_voc should be a list of sub-lists
    of strings containing all the sentences tokenized.
    The output is a single list of strings containing 
    all the unique vocabulary tokenized words.

    @given:
    ========
    list_words: list[str]
                Sequence containing the sentences from which to extrapolate
                the vocabulary.
    '''
    # get first the vocabulary
    sentences = get_vocabulary(list_words)

    # sentences is a list of lists of strings: flatten the sequence 
    # and find only the unique tokens
    voc_unique = flatten_unique_voc(sentences)

    word_tokenize_uniq = [nltk.word_tokenize(word) for word in list_words]
    flat_list = [item for sublist in word_tokenize_uniq for item in sublist]
    uniq_words = " ".join(set(flat_list))
    # control that the dimensionality of voc_unique is equal to the 
    # dimensionality of split uniq_words
    assert(len(uniq_words.split()) ==  len(voc_unique))

def test_max_len():
    '''
    Test function to test the behaviour of calculate_max_len function.
    The input to calculate_max_len should be a list of sentences that the 
    user wants to compute the maximum sentence length.
    The output is the maximum vector length to pad/truncate sequences
    for the neural network model.
    The max length is equal to the mean of the word counts + 
    2*standard deviation of the word counts.
    '''
    random_text = ['another one bites the dust', 'another string', 'word',
                    'four characters for me', 'awful string']

    maxlen = calculate_max_len(random_text)
    # the mean of the word counts in random_text
    # is 2.8 and the standard deviation is 1.47
    # so we expect that maxlen is equal to 6 (2.8 + 1.47*2)
    assert(maxlen == 6)

def test_encode_labels():
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    The function takes as input the labels, that will be encoded in 
    integer numbers.
    The output is an numpy array with the same size of the initial list
    composed by numbers from 0 to n-1 where n is the number of the classes.
    The function returns also the class neames associated to the numbers.
    '''
    unique_labels = ['real', 'fake']
    # create fake labels
    labels = [unique_labels[random.randrange(len(unique_labels))] for _ in range(100)]
    # find indices regarding the two labels
    indices = {'real': [index for (index, label) in enumerate(labels) if label == 'real'],
               'fake': [index for (index, label) in enumerate(labels) if label == 'fake']}
    # convert labels
    y_categorical, classes = tocat_encode_labels(labels, classes = True)

    # original labels ('real', 'fake') in total were 100
    # also categorical labels have to be 100:
    assert(len(y_categorical) == 100)
    # the classes are the unique labels
    assert(set(classes) == {'real', 'fake'})
    # the type of encoded labels are np array of integers (in this case 0, 1)
    assert(type(y_categorical) == np.ndarray)
    assert(set(y_categorical) == {0, 1})

    # find the indices in the categorical labels
    y_categorical_0index = np.where(y_categorical == 0)
    y_categorical_1index = np.where(y_categorical == 1)

    # verify that inidices of the categorical labels (0, 1) are equal
    # to the indices of original labels ('real', 'fake')
    assert((np.array(indices[classes[0]]) == y_categorical_0index).all())
    assert((np.array(indices[classes[1]]) == y_categorical_1index).all())

@given(unique_labels = st.lists(st.sampled_from(string.ascii_letters),
                                 min_size = 1, max_size=10, unique=True))
def test_encode_random_labels(unique_labels):
    '''
    Test function to test the behaviour of tocat_encode_labels function
    for multiple labels (not just two).
    tocat_encode_labels takes as input the labels, that will be encoded
    in integer numbers. The output is an numpy array with the same size
    of the initial list composed by integer numbers from 0 to n-1 
    where n is the number of the classes.
    The function returns also the class names associated to the numbers.

    @given:
    unique_labels: list[str]
                   Sequence of strings that correspond to the unique labels.
    '''
    # generate some random labels
    labels_list = [unique_labels[random.randrange(len(unique_labels))] 
                                                    for _ in range(500)]
    indices = {}
    # find the indices for each label
    for label in unique_labels:
        indices[label] = [index for (index, lab) in enumerate(labels_list)
                                                             if label == lab]
    # convert the labels and return also classes 
    y_categorical, classes = tocat_encode_labels(labels_list, classes=True)
    
    # original labels in total were 500
    # also categorical labels have to be 500:
    assert(len(y_categorical) == 500)
    # the classes set is a subset of the unique labels
    assert(set(classes).issubset(set(unique_labels)))
    
    # the type of encoded labels are np array of integers
    assert(type(y_categorical) == np.ndarray)
    for y_cat in y_categorical:
        assert(type(y_cat.item()) == int)

    y_indices = {}
    # find indices for encoded labels
    for y_cat in np.unique(y_categorical):
        y_indices[y_cat] = np.where(y_categorical == y_cat) 

    # verify that inidices of the categorical labels are equal
    # to the indices of original labels
    for class_, y_cat in zip(classes, np.unique(y_categorical)):
        assert((np.array(indices[class_]) == y_indices[y_cat]).all())


def test_shape_mean_emb_vect():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    The MeanEmbeddingVectorizer class makes an average of the vector words
    in the sentence passed to the transformer. The vector words are passed 
    to the class by a dictionary. The dictionary is the result
    of a Word2Vec model.

    In this test function the shape is tested: the output shape should be
    (number of sentences, embedding vector size)
    '''
    # some random text for the vocabulary
    random_text_voc = ['hello world', 'hello my name is Tom',
                    'Tom loves catching fishes', 'random word in a random string',
                    'hello what a beautiful day',
                    'the sky is blue']
    # find vocabulary
    vocabulary = get_vocabulary(random_text_voc)
    
    embedding_vector_size = 10 # output vector size
    min_count_words = 2 # minimum number of occurencies for words (Word2Vec model)
    random_state = 42 # seed

    # create the Word2Vec model
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5,
                         min_count=min_count_words, workers=1, seed = random_state)   
    # dictionary that maps each word inside the Word2Vec model to the corresponding
    # vector
    w2v = dict(zip(modelw2v.wv.index_to_key, modelw2v.wv.vectors)) 
    
    # text to transform
    text_to_transform = ['hello world, my name is Flavio',
                         'in my free time i love catching fishes']
    # tokenization of the text
    X_tok = [nltk.word_tokenize(words) for words in text_to_transform]  

    mean_emb_vect = MeanEmbeddingVectorizer(w2v)
    X_vectors_w2v = mean_emb_vect.transform(X_tok)
    assert(np.shape(X_vectors_w2v) == 
                        (len(text_to_transform), embedding_vector_size)) 


def test_value_mean_emb_vect():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    The MeanEmbeddingVectorizer class makes an average of the vector words
    in the sentence passed to the transformer. The vector words are passed 
    to the class by a dictionary. The dictionary is the result
    of a Word2Vec model.
    
    In this test function the correct values of sentences passed to 
    MeanEmbeddingVectorizer are tested: the output should be the average of the word
    vectors contained in the Word2Vec dictionary, or a zeros array if there is
    no match between words from sentence and words from dictionary.

    In this test function the minimun count words is set to 1: it means
    that all the words from the vocabulary will be part of the Word2Vector dictionary.
    If the minimun count words is set to 2 (for example): it means that only the words
    with at least two occurencies from the vocabulary will be part of the Word2Vec
    dictionary. 
    '''
    # random text fot the vocabulary
    random_text_voc = ['hello world', 'hello my name is Tom',
                    'Tom loves catching fishes', 'random word in a random string',
                    'hello what a beautiful day',
                    'the sky is blue']
    # get the vocabulary
    vocabulary = get_vocabulary(random_text_voc)
    embedding_vector_size = 10 # output vector size
    min_count_words = 1 # minimum number of occurencies for words (Word2Vec model)
    random_state = 42 # seed

    # Word2Vec model
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5,
                         min_count=min_count_words, workers=1, seed = random_state)   
    # dictionary that maps each word inside the Word2Vec model to the corresponding
    # vector:
    w2v = dict(zip(modelw2v.wv.index_to_key, modelw2v.wv.vectors)) 
    
    # some text to transform
    text_to_transform = ['hello',
                         'Flavio',
                         "hello world my name is Tom and I'm 20 years old"]
    # tokenization of text
    X_tok = [nltk.word_tokenize(words) for words in text_to_transform]  

    mean_emb_vect = MeanEmbeddingVectorizer(w2v)
    X_vectors_w2v = mean_emb_vect.transform(X_tok)
    
    # the first vector represents 'hello'
    assert( np.isclose(X_vectors_w2v[0], w2v['hello']).all() )

    # in the second sentence there is no match between dictionary words and 
    # sentence words. so the return output is a zeros vector:
    assert( (X_vectors_w2v[1] == np.zeros(shape=(1, embedding_vector_size))).all() )

    # the third sentence is composed by words in the dictionary
    # so we compute the average of the word vectors present both in the sentence and
    # in the dictionary:
    assert( np.isclose(X_vectors_w2v[2], np.mean([w2v['hello'], w2v['world'], w2v['my'],
                                        w2v['name'], w2v['is'], w2v['Tom']], axis = 0)).all() )

def test_vect_X_lr():
    '''
    Test function to test the behaviour of vectorize_X_data_lr function.

    vectorize_X_data_lr takes as input the text the user wants to vectorize, which 
    is a list of strings and a model of Word2Vec already trained.

    The output is the vectorized text.
    '''
    # text for vocabulary
    random_text_voc = ['hello world', 'hello my name is Tom',
                    'Tom loves catching fishes', 'random word in a random string',
                    'hello what a beautiful day',
                    'the sky is blue']
    # get vocabulary:
    vocabulary = get_vocabulary(random_text_voc)
    
    embedding_vector_size = 10 # output vector size
    min_count_words = 1 # minimum number of occurencies for words (Word2Vec model)
    random_state = 42 # seed

    # create the Word2Vec model:
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5,
                         min_count=min_count_words, workers=1, seed = random_state)   
    # some text to vectorize:
    text_to_vectorize= ['hello',
                         'Flavio',
                         'hello world my name is Tom',
                         "I'm Flavio, despite Tom I don't catch fishes"]
    
    # vectorize the text
    X_vector = vectorize_X_data_lr(text_to_vectorize, modelw2v)
    
    # dictionary useful for the mapping of words and vectors
    w2v = dict(zip(modelw2v.wv.index_to_key, modelw2v.wv.vectors)) 

    # first sentence is composed only by 'hello'
    assert( np.isclose(X_vector[0], w2v['hello']).all() )
    
    # second sentence is composed by words not present in the 
    # vocabulary -> zeros array
    zero_array = np.zeros(shape=(embedding_vector_size))
    assert( (X_vector[1] ==  zero_array).all() )

    # third sentence is composed by words present in the vocabulary
    mean_third_phrase = np.mean([w2v['hello'],  w2v['my'],
                                w2v['world'], w2v['name'],    
                                w2v['is'],w2v['Tom']], axis = 0)
    assert( np.isclose(X_vector[2], mean_third_phrase).all() )

    # fourth sentence is composed by some words present in the vocabulary
    # it is computed the average between these word vectors
    mean_fourth_sentence = np.mean([w2v['fishes'], w2v['Tom']], axis = 0)
    assert( np.isclose(X_vector[3], mean_fourth_sentence).all() )

@given(embedding_vector_size = st.integers(min_value = 1, max_value = 40),
       text_to_vectorize = st.lists(st.sampled_from(('hello', 'world', 
                                            ''.join(string.ascii_letters))), 
                                    min_size = 1, max_size = 40))
def test_shape_vect_X_lr(embedding_vector_size, text_to_vectorize):
    '''
    Test function to test the behaviour of vectorize_X_data_lr function.
    In particular, this test function tests the correct output shape for the 
    vectorize_X_data_lr function with some random text and random vector size.
    vectorize_X_data_lr takes as input the text the user wants to vectorize, which 
    is a list of strings and a model of Word2Vec already trained.
    The output is the vectorized text with a precise shape, that
    is given by (number of strings in the initial list, embedding vector size).

    @given:
    embedding_vector_size: int
                           Integer number that represents the output vector size.
    
    text_to_vectorize: list[str]
                       Sequence of text that represents the text to vectorize.
    '''
    # random text to get vocabulary
    random_text_voc = ['hello world', 'hello my name is Tom',
                    'Tom loves catching fishes', 'random word in a random string',
                    'hello what a beautiful day',
                    'the sky is blue']
    vocabulary = get_vocabulary(random_text_voc)
    
    min_count_words = 1 # minimum number of occurencies for words (Word2Vec model)
    random_state = 42 # seed

    # Word2Vec model
    modelw2v = Word2Vec(vocabulary, vector_size=embedding_vector_size, window=5,
                         min_count=min_count_words, workers=1, seed = random_state)   
    
    X_vector = vectorize_X_data_lr(text_to_vectorize, modelw2v)
    
    assert( X_vector.shape == (len(text_to_vectorize), embedding_vector_size))

def test_vector_layer():
    '''
    Test function to test the behaviour of init_vector_layer function.
    '''
    # some random text to get vocabulary
    random_text_voc = ['hello world', 'hello my name is Tom',
                    'Tom loves catching fishes', 'random word in a random string',
                    'hello what a beautiful day',
                    'the sky is blue']
    # get the vocabulary: output is a list[list[str]]
    vocabulary = get_vocabulary(random_text_voc)

    # for TextVectorization layer we need list[str], not list[list[str]], 
    # so we flatten the vocabulary and take only the unique tokens
    uniq_voc = flatten_unique_voc(vocabulary)

    # we calcualte maxlen to pad/truncate sequences
    maxlen = calculate_max_len(random_text_voc)

    # vector layer initialization:
    vector_layer = init_vector_layer(maxlen, uniq_voc)
    
    # we get from the layer the vocabulary
    voc_from_layer = vector_layer.get_vocabulary(include_special_tokens = False)
    voc_size = vector_layer.vocabulary_size()

    # all the words passed from the beginning vocabulary are present 
    # in the layer vocabulary
    assert((uniq_voc == voc_from_layer).all())
    
    # the sizes of the different vocabulary match 
    assert(len(uniq_voc) + 2 == voc_size) #+2 because of special tokes
    
    # control type of vector layer:
    assert(type(vector_layer) == tf.keras.layers.TextVectorization)


@given(maxlen = st.integers(min_value = 1, max_value = 40),
       text_to_vectorize = st.lists(st.sampled_from(('hello', 'world', 'Tom', 'random',
                            ''.join(string.ascii_letters))), min_size = 1, max_size = 40))
def test_shape_vectorize_X_tf(maxlen, text_to_vectorize):
    '''
    Test function to test the behaviour of vectorize_X_data_tf function.
    In particular, this test function tests the correct output shape of vectors.
    vectorize_X_data_tf takes as input the text the user wants to vectorize, which 
    is a list of strings (or a string) and the tensorflow vectorization layer.

    The output is the vectorized text with a precise shape, that
    is given by (number of sentences, maximum vector length).
    Eventually some padding/truncation operations are
    performed by the layer to have arrays with the same size.

    @given:
    ========
    maxlen: int
            Integer number that corresponds to the maximum vector length
    
    text_to_vectorize: list[str]
                       Random text to vectorize.
    '''
    # some random text to get vocabulary
    random_text_voc = ['hello world', 'hello my name is Tom',
                    'Tom loves catching fishes', 'random word in a random string',
                    'hello what a beautiful day',
                    'the sky is blue']
    vocabulary = get_vocabulary(random_text_voc)
    # find unique and flat vocabulary
    uniq_voc = flatten_unique_voc(vocabulary)

    vector_layer = init_vector_layer(maxlen, uniq_voc)

    #vector = tf.stack([vectorize_X_data_tf(text, vector_layer) for text in text_to_vectorize])
    vector = vectorize_X_data_tf(text_to_vectorize, vector_layer)
    assert(vector.shape == (len(text_to_vectorize), maxlen))

def test_value_vectorize_X_tf():
    '''
    Test function to test the behavior of vectorize_X_data_tf function.
    In this test function the correct output values of the text
    after the vectorization are tested.
    vectorize_X_data_tf takes as input the text the user wants to vectorize, which 
    is a list of strings and the tensorflow vectorization layer.
    The output is the vectorized text. 

    If there is a match between the words to vectorize and the words
    stored in the vocabulary, the word is represented by the integer index number
    of the word in the vocabulary.
    
    If the word in the text to vectorize is not present in the vocabulary,
    the vectorization layer represents this word with "1".

    At the end of the sentences there could be some zeros to pad the sequences.
    '''
    # random text to get vocabulary
    random_text_voc = ['hello world', 'hello my name is Tom',
                    'Tom loves catching fishes', 'random word in a random string',
                    'hello what a beautiful day',
                    'the sky is blue']
    vocabulary = get_vocabulary(random_text_voc)
    uniq_voc = flatten_unique_voc(vocabulary)
    # compute maximum vector length
    maxlen = calculate_max_len(random_text_voc)

    vector_layer = init_vector_layer(maxlen, uniq_voc)

    # random text to vectorize
    text_to_vectorize = ['hello',
                        'my name is Flavio',
                        '33 trentini']

    vector = vectorize_X_data_tf(text_to_vectorize, vector_layer)

    assert(vector.dtype == tf.int32) # the type is int 32 bit

    # now check the value assigned to each word
    complete_voc = vector_layer.get_vocabulary()

    # we expect 'hello' in vector (vector[0][0])
    # is equal to the index of the 'hello' word in the vocabulary from the layer 
    assert(vector[0][0] == complete_voc.index('hello'))

    # we expect that vector[0] is equal to = index('hello'),0,0,0,0,0,0
    # where the zeros are giving by the padding operation (in this exampe there are
    # maxlen - 1 zeros, because there is just a single word)
    hello_in_indices = np.array(complete_voc.index('hello'))
    hello_in_indices = hello_in_indices.reshape((1,))
    zero_array = np.zeros(dtype = int, shape = (maxlen - hello_in_indices.size))

    # create the vector that represents the complete sentence: index of 'hello', followed
    # by a zero arrays of size maxlen - 1
    hello_in_indices = np.concatenate((hello_in_indices, zero_array), axis = 0)
    # verify that the first sentence is vectorized as expected
    assert((np.array(vector[0]) == hello_in_indices).all())

    # second sentece: find the indices for the words
    indices_vector_two = np.array([complete_voc.index('my'),
               complete_voc.index('name'),
               complete_voc.index('is'),
               1]) # the one is for the 'Flavio' word (which is not inside the vocabulary)
    # pad the sequence with a zeros array of size maxlen - 4
    zero_array = np.zeros(dtype = int, shape = (maxlen - indices_vector_two.size))
    # vector that represents completely the second sentence:
    indices_vector_two = np.concatenate((indices_vector_two, zero_array), axis = 0)
    # verify that the second sentence is vectorized as expected
    assert((np.array(vector[1]) == indices_vector_two).all())

    # '33' 'trentini' are not words present in the vocabulary -> 1 to represent them
    indices_vector_three = np.array([1, 1]) 
    # pad the sequence with a zeros array of size maxlen - 2
    zero_array = np.zeros(dtype = int, shape = (maxlen - indices_vector_three.size,))
    # vector that represents completely the third sentence:
    indices_vector_three = np.concatenate((indices_vector_three, zero_array), axis = 0)
    # verify that the third sentence is vectorized as expected
    assert((np.array(vector[2]) == indices_vector_three).all())
