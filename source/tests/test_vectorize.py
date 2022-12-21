'''
Test functions for testing the functions inside the vectorize_data module.
'''
from text_analysis.vectorize_data import vectorize_X_data_tf
from text_analysis.vectorize_data import calculate_max_len, get_vocabulary
from text_analysis.vectorize_data import MeanEmbeddingVectorizer
from text_analysis.vectorize_data import init_vector_layer, tocat_encode_labels
from text_analysis.vectorize_data import flatten_unique_voc
import numpy as np
from hypothesis import strategies as st
from hypothesis import given, settings

def test_number_of_sentences_from_vocabulary():
    '''
    Test function to test the behaviour of get_vocabulary function.
    In this test function it's tested that the number of sentences 
    tokenized is correct.

    Given:
    ======
    two_sentences: list[str]
                   Two sentences from which to extrapolate the vocabulary.
    
    Tests:
    ======
            if the number of tokenized sentences is correct (equal to 2).
    '''
    two_sentences = ['hello world', 'what a beautiful day']
    sentences_tokenized = get_vocabulary(two_sentences)
    assert(len(sentences_tokenized) == 2) 

def test_number_of_tokens_from_vocabulary():
    '''
    Test function to test the behaviour of get_vocabulary function.
    In this test function it's tested if the number of tokenized
    words is correct.

    Given:
    ======
    sentence_two_words: list[str]
                        Input sentence with two words 
                        from which to extrapolate the vocabulary.
    
    Tests:
    ======
            if the number of tokens from the sentence is correct (two words).
    '''
    sentence_two_words = ['hello world']
    sentences_tokenized = get_vocabulary(sentence_two_words)
    assert(len(sentences_tokenized[0]) == 2)


def test_vocabulary_no_multiple_occurencies_single_sentence():
    '''
    Test function to test the behaviour of get_vocabulary function.
    In this test function it's tested what's the result of the function
    if a single sentence that doesn't contain any repeated words is passed
    as input parameter.

    Given:
    ======
    sentences: list[str]
               Input sentence from which to extrapolate the vocabulary.
    
    Tests:
    ======
            if all the words from the single sentence are part of the 
            vocabulary.
    '''

    sentences = ['hello world i am flavio']
    sentences_tokenized = get_vocabulary(sentences)
    assert(sentences_tokenized == [['hello', 'world', 'i', 'am', 'flavio']])

def test_vocabulary_no_multiple_occurencies_two_sentences():
    '''
    Test function to test the behaviour of get_vocabulary function.
    In this test function it's tested what's the result of the function
    if two sentences that don't contain any repeated words are passed
    as input parameter.
    
    Given:
    ======
    sentences: list[str]
               Input sentences from which to extrapolate the vocabulary.
    
    Tests:
    ======
            if all the words from both the two sentences are part of the 
            vocabulary.
    '''

    sentences = ['hello world', 'i am flavio']
    sentences_tokenized = get_vocabulary(sentences)
    assert(sentences_tokenized == [['hello', 'world'], ['i', 'am', 'flavio']])

def test_vocabulary_if_multiple_occurencies_in_different_sentences():
    '''
    Test function to test the behaviour of get_vocabulary function.
    In this test function it's tested what's the result of the function
    if two sentences that contain some repeated words are passed
    as input parameter.
    
    Given:
    ======
    sentences: list[str]
               Input sentences from which to extrapolate the vocabulary.
    
    Tests:
    ======
            if all words from both the two sentences are part of the 
            vocabulary, even though there are repeated words.
    '''
    sentences = ['hello world', 'hello world i am flavio']
    sentences_tokenized = get_vocabulary(sentences)
    assert(sentences_tokenized == [['hello','world'], ['hello','world','i','am','flavio']])

def test_vocabulary_if_multiple_occurencies_in_same_sentences():
    '''
    Test function to test the behaviour of get_vocabulary function.
    In this test function it's tested what's the result of the function
    if a single sentence that contains some repeated words is passed
    as input parameter.
    
    Given:
    ======
    sentence_with_repeated_words: list[str]
                                  Input sentence with repeated words 
                                  from which to extrapolate the vocabulary.
    
    Tests:
    ======
            if all words from the single sentence are part of the 
            vocabulary, even though there are repeated words.
    '''
    sentences = ['hello world, hello world']
    sentences_tokenized = get_vocabulary(sentences)
    assert(sentences_tokenized == [['hello','world',',','hello','world']])

def test_get_vocabulary_with_empty_string():
    '''
    Test function to test the behaviour of get_vocabulary function.
    In this test function it's tested what's the result of the function
    if a an empty string is passed to the function as input parameter.
    
    Given:
    ======
    sentences_with_empty_string: list[str]
                                 List composed only by an empty string.
    
    Tests:
    ======
            if the vocabulary is composed by an empty list, since
            it is not passed any words.
    '''
    sentences_with_empty_string = ['']
    voc_from_empty_string = get_vocabulary(sentences_with_empty_string)
    assert(voc_from_empty_string == [[]])


def test_flatten_vocabulary():
    '''
    Test function to test the behaviour of flatten_unique_voc function.
    In this test function it's tested what's the result of the function
    if two tokenized sentences without repeated words
    are passed as input parameter.
    
    Given:
    ======
    vocabulary: list[list[str]]
                List composed by tokenized sentences without repeated words.
    
    Tests:
    ======
            if the flattened vocabulary is composed by all the words present in the 
            tokenized sentences, since there aren't repeated words.
    '''
    vocabulary = [['hello', 'world'], ['i', 'am', 'flavio']]
    flattened_vocabulary = flatten_unique_voc(vocabulary)
    assert(set(flattened_vocabulary) == {'hello', 'world', 'i', 'am', 'flavio'})
    
def test_unique_vocabulary_from_two_sentences():
    '''
    Test function to test the behaviour of flatten_unique_voc function.
    In this test function it's tested what's the result of the function
    if two tokenized sentences with repeated words are passed as input
    parameter.
    
    Given:
    ======
    vocabulary: list[list[str]]
                List composed by tokenized sentences with some repeated words.
    
    Tests:
    ======
            if the flattened vocabulary is composed only by the unique 
            words present in the tokenized sentences.
    '''
    vocabulary = [['hello', 'world'], ['hello', 'world']]
    unique_vocabulary = flatten_unique_voc(vocabulary)
    assert(list(unique_vocabulary) == ['hello', 'world'])

def test_unique_vocabulary_from_single_sentence():
    '''
    Test function to test the behaviour of flatten_unique_voc function.
    In this test function it's tested what's the result of the function
    if a single tokenized sentence with repeated words is passed as input
    parameter.
    
    Given:
    ======
    vocabulary: list[list[str]]
                List composed by a single tokenized sentence
                with some repeated words.
    
    Tests:
    ======
            if the flattened vocabulary is composed only by the unique 
            words present in the tokenized sentence.
    '''
    vocabulary = [['hello', 'world', 'hello', 'world']]
    unique_vocabulary = flatten_unique_voc(vocabulary)
    assert(list(unique_vocabulary) == ['hello', 'world'])

def test_flatten_vocabulary_with_empty_string():
    '''
    Test function to test the behaviour of flatten_unique_voc function.
    In this test function it's tested what's the result of the function
    if an empty string is passed as input parameter.
    
    Given:
    ======
    empty_string_vocabulary: list[list[str]]
                             List composed only by a single list
                             containing an empty string.
    
    Tests:
    ======
            if the list is flattened: we obtain only a single list
            with an empty string ('').
    '''
    empty_string_vocabulary = [['']]
    flatten_empty_string = flatten_unique_voc(empty_string_vocabulary)
    assert(list(flatten_empty_string) == [''])

def test_flatten_vocabulary_with_empty_list():
    '''
    Test function to test the behaviour of flatten_unique_voc function.
    In this test function it's tested what's the result of the function
    if an empty list is passed as input parameter.
    
    Given:
    ======
    empty_list_vocabulary: list[list]
                           List composed by empty list.
    
    Tests:
    ======
            if the list is flattened: we obtain only a single list
            and anymore a list of list.
    '''
    empty_list_vocabulary = [[]]
    flatten_empty_list = flatten_unique_voc(empty_list_vocabulary)
    assert(list(flatten_empty_list) == [])

def test_max_len():
    '''
    Test function to test the behaviour of calculate_max_len function.
    In this test function it's tested what's the result of the function
    if it is passed three sentence with length equal to: 1, 2, 3. 
    The average of the number words is equal to 2, while the standard
    deviation of the number of words is equal to 1.
    We expect the output is equal to 4.

    Given:
    =======
    sentences: list[str]
               Input sentences from which we want to compute
               the maximum length (average + 2*std).

    Tests:
    ======
            if sentences is composed by 1, 2 and 3 words:
            maximum_length should be equal to 4, since
            average number of words is 2 and std of number
            of words is 1.
    '''
    sentences = ['two words', 'word', 'i am flavio']
    maximum_length = calculate_max_len(sentences)
    assert(maximum_length == 4)

def test_single_label():
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    In this test function it's tested what's the result if a single 
    class label is passed to the function.

    Given:
    ======
    labels: list[str]
            List of all the labels: only a single class.
    
    Tests:
    ======
            if all the categorical encoded labels are 
            equal to 0 (only a single label class).
    '''
    labels = ['a', 'a', 'a']
    y_categorical = tocat_encode_labels(labels)
    assert(set(y_categorical) == {0})

def test_two_labels():
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    In this test function it's tested what's the result if two 
    class labels are passed to the function.

    Given:
    ======
    labels: list[str]
            List of all the labels: two different classes.
    
    Tests:
    ======
            if the categorical encoded labels are 
            equal to 0 or 1 (two classes).
    '''
    labels = ['a', 'b', 'b', 'a']
    y_categorical = tocat_encode_labels(labels)
    assert(set(y_categorical) == {0, 1})

def test_three_labels():
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    In this test function it's tested what's the result if three 
    class labeles are passed to the function.

    Given:
    ======
    labels: list[str]
            List of all the labels: three different classes.
    
    Tests:
    ======
            if the categorical encoded labels are 
            equal to 0, 1 or 2 (three classes).
    '''
    labels = ['a', 'b', 'c', 'b', 'a']
    y_categorical = tocat_encode_labels(labels)
    assert(set(y_categorical) == {0, 1, 2})

def test_index_two_labels():
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    In this test function it's tested if the categorical encoded labels
    correctly corresponds to the original labels.

    Given:
    ======
    labels: list[str]
            List of all the labels: two different classes.
    
    Tests:
    ======
            if the categorical encoded labels are correctly
            oredered (so for each 'a' we expect a 0 and for 
            each 'b' we expect a 1).
    '''
    labels = ['a', 'a', 'b', 'a', 'b']
    y_categorical = tocat_encode_labels(labels)
    assert(list(y_categorical) == [0, 0, 1, 0, 1])

@given(labels = st.lists(st.floats(), min_size = 1, max_size = 1))
def test_floats_as_input_labels(labels):
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    In this test function it's tested if a float label can fit the 
    function as parameter.

    Given:
    ======
    labels: list[float]
            List of all the labels: float labels in this test case.
    
    Tests:
    ======
            if the function tocat_encode_labels can work with 
            float labels as input. 
    '''
    y_categorical = tocat_encode_labels(labels)
    assert(set(y_categorical) == {0})

@given(labels = st.lists(st.integers(), min_size = 1, max_size = 1))
def test_integers_as_input_labels(labels):
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    In this test function it's tested if an integer label can fit the 
    function as parameter.

    Given:
    ======
    labels: list[float]
            List of all the labels: integer labels in this test case.
    
    Tests:
    ======
            if the function tocat_encode_labels can work with 
            integer labels as input. 
    '''
    y_categorical = tocat_encode_labels(labels)
    assert(set(y_categorical) == {0})

def test_classes_single_label():
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    In this test function it's tested what's the classes value
    if only a single class label are passed as input parameter.
    We expect to have only the original single class.

    Given:
    ======
    labels: list[float]
            List of all the labels with only a single class.
    
    Tests:
    ======
            if classes is composed only by the single class
            of the labels.
    '''
    labels = ['a', 'a']
    y_categorical, classes = tocat_encode_labels(labels, classes = True)
    assert(classes == ['a'])

def test_classes_two_labels():
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    In this test function it's tested what's the classes value
    if only two class labels are passed as input parameter. 
    We expect to have only two different classes (the same as the 
    original class labels).

    Given:
    ======
    labels: list[float]
            List of all the labels with two classes.
    
    Tests:
    ======
            if classes is composed only by the the two classes
            of the labels.
    '''
    labels = ['a', 'b', 'a', 'b']
    y_categorical, classes = tocat_encode_labels(labels, classes = True)
    assert(list(classes) == ['a', 'b'])

def test_classes_three_labels():
    '''
    Test function to test the behaviour of tocat_encode_labels function.
    In this test function it's tested what's the classes value
    if only three class labels are passed as input parameter. 
    We expect to have only three different classes (the same as the 
    original class labels).

    Given:
    ======
    labels: list[float]
            List of all the labels with three classes.
    
    Tests:
    ======
            if classes is composed only by the the three classes
            of the labels.
    '''
    labels = ['a', 'b', 'c', 'a', 'b']
    y_categorical, classes = tocat_encode_labels(labels, classes = True)
    assert(list(classes) == ['a', 'b', 'c'])


def test_shape_mean_embedding_vectorizer_single_sentence_with_no_mapping_in_vocabulary():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    In this test function it's tested that the output shape is correct if we pass
    a single sentence to vectorize with no word mapping in the vocabulary.

    Given:
    ======
    tokenized_sentences: list[list[str]]
                         The list contains the tokenized sentences (list of strings).
                         In this case only a single sentence is used.
    
    Tests:
    ======
            if the vectorized text has a shape consistent with one single
            sentence passed (in this case the shape is (1, 0) because we have a single 
            sentence and an embedding vector size equal to zero).
    '''
    tokenized_sentences = [['hello', 'world']]
    mean_emb_vect = MeanEmbeddingVectorizer({'': []}) # embedding vector size equal to zero, since empty list
    X_vectors_w2v = mean_emb_vect.transform(tokenized_sentences)
    assert(X_vectors_w2v.shape == (1, 0))

def test_shape_mean_embedding_vectorizer_three_sentences_with_no_mapping_in_vocabulary():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    In this test function it's tested that the output shape is correct if we pass
    three sentences to vectorize with no word mapping in the vocabulary.

    Given:
    ======
    tokenized_sentences: list[list[str]]
                         The list contains the tokenized sentences (list of strings).
                         In this case three sentences are passed.
    
    Tests:
    ======
            if the vectorized text has a shape consistent with three
            sentences passed (in this case the shape is (3, 0) because we have three 
            sentences and an embedding vector size equal to zero).
    '''
    tokenized_sentences = [['hello', 'world'], ['i', 'am', 'flavio'], ['how', 'are', 'you', '?']]
    mean_emb_vect = MeanEmbeddingVectorizer({'': []}) # embedding vector size equal to zero, since empty list
    X_vectors_w2v = mean_emb_vect.transform(tokenized_sentences)
    assert(X_vectors_w2v.shape == (3, 0))

def test_shape_mean_embedding_vectorizer_single_sentence_with_vocabulary():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    In this test function it's tested that the output shape is correct if we pass
    a single sentence to vectorize with complete word mapping in the vocabulary.

    Given:
    ======
    tokenized_sentences: list[list[str]]
                         The list contains the tokenized sentences (list of strings).
                         In this case only a single sentence is used, with a complete
                         mapping between words in the sentence and words in the 
                         vocabulary.
    
    dictionary_mapping_word_to_vec: dict
                                    Dictionary that maps the words (tokens) to a 
                                    number vector.
    
    Tests:
    ======
            if the vectorized text has a shape consistent with a single
            sentence passed (the output shape should be equal to (1,2), since 
            we pass a single sentence and we use an embedding vector size equal to 2).
    '''
    tokenized_sentences = [['hello', 'world']]
    dictionary_mapping_word_to_vec = {'hello': [0, 1], 'world': [1, 0]} # embedding vector size equal to two
    mean_emb_vect = MeanEmbeddingVectorizer(dictionary_mapping_word_to_vec)
    X_vectors_w2v = mean_emb_vect.transform(tokenized_sentences)
    assert(X_vectors_w2v.shape == (1, 2))

def test_shape_mean_embedding_vectorizer_three_sentences_with_vocabulary():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    In this test function it's tested that the output shape is correct if we pass
    three sentences to vectorize with partial word mapping in the vocabulary.

    Given:
    ======
    tokenized_sentences: list[list[str]]
                         The list contains the tokenized sentences (list of strings).
                         In this case only three sentences are used, with a partial
                         mapping between words in the sentence and words in the 
                         vocabulary.
    
    dictionary_mapping_word_to_vec: dict
                                    Dictionary that maps the words (tokens) to a 
                                    number vector.
    
    Tests:
    ======
            if the vectorized text has a shape consistent with a three
            sentences passed (in this case the output shape should be equal to (3, 2), since
            we use three sentences and an embedding vector size equal to two).
    '''
    tokenized_sentences = [['hello', 'world'], ['i', 'am', 'flavio'], ['how', 'are', 'you', '?']]
    dictionary_mapping_word_to_vec = {'hello': [0, 1], 'world': [1, 0]} # embedding vector size equal to 2
    mean_emb_vect = MeanEmbeddingVectorizer(dictionary_mapping_word_to_vec)
    X_vectors_w2v = mean_emb_vect.transform(tokenized_sentences)
    assert(X_vectors_w2v.shape == (3, 2))

def test_shape_mean_embedding_vectorizer_embedding_vector_size_equal_to_four():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    In this test function it's tested that the output shape is correct if we want
    to obtain an output vector with embedding vector size equal to 4.

    Given:
    ======
    tokenized_sentences: list[list[str]]
                         The list contains the tokenized sentences (list of strings).
                         In this case only it is used only a single sentence.

    dictionary_with_4_vector_size: dict
                                   Dictionary that maps each single word to a vector
                                   of dimension equal to 4.
    
    Tests:
    ======
            if the vectorized text has a shape equal to (1, 4), since 
            we pass only a single sentence and we choose an embedding 
            vector size equal to 4 (the vector size is the length 
            of the vectors that map the words).
    '''
    tokenized_sentences = [['hello', 'world']]
    dictionary_with_4_vector_size = {'': [0, 0, 0, 0]} # embedding vector size equal to 4
    mean_emb_vect = MeanEmbeddingVectorizer(dictionary_with_4_vector_size)
    X_vectors_w2v = mean_emb_vect.transform(tokenized_sentences)
    assert(X_vectors_w2v.shape == (1, 4))


def test_value_mean_embedding_vectorizer_with_no_mapping_in_vocabulary():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    In this test function it's tested what's the result if we pass to the
    transform function a sentence composed only by words without mapping
    in the vocabulary. The output should be a vector of length equal to
    the vector size composed only by zeros.

    Given:
    ======
    tokenized_sentences: list[list[str]]
                         The list contains the tokenized sentences (list of strings).
                         In this case it is used only a single sentence without
                         mapping between words from sentences and words in the
                         vocabulary.

    dictionary_mapping_word_to_vec: dict
                                    Dictionary that maps each single word to a vector
                                    of dimension equal to 3.
    
    Tests:
    ======
            if the result is an array of dimension 3 (vector size is equal to three)
            composed by only zeros value.
    '''
    tokenized_sentences = [['hello', 'world']]
    dictionary_mapping_word_to_vec = {'': [0, 0, 1]} # embedding vector size is equal to three
    mean_emb_vect = MeanEmbeddingVectorizer(dictionary_mapping_word_to_vec)
    X_vectors_w2v = mean_emb_vect.transform(tokenized_sentences)
    assert(list(X_vectors_w2v[0]) == [0, 0, 0])

def test_value_mean_embedding_vectorizer_with_complete_mapping_in_vocabulary():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    In this test function it's tested what's the result if we pass to the
    transform function a sentence composed only by words with complete mapping
    in the vocabulary. The output should be a vector of length equal to
    the embedding vector size, composed by the average of the word vectors.

    Given:
    ======
    tokenized_sentences: list[list[str]]
                         The list contains the tokenized sentences (list of strings).
                         In this case only it is used only a single sentence with
                         complete mapping between words from sentence and words in the
                         vocabulary.

    dictionary_mapping_word_to_vec: dict
                                    Dictionary that maps each single word to a vector
                                    of dimension equal to 2.
    
    Tests:
    ======
            if the result is an array of dimension 2 (vector size is equal to two)
            composed by the average value of the word vectors.
    '''
    tokenized_sentences = [['hello', 'world']]
    dictionary_mapping_word_to_vec = {'hello': [0, 1], 'world': [1, 0]}
    mean_emb_vect = MeanEmbeddingVectorizer(dictionary_mapping_word_to_vec)
    X_vectors_w2v = mean_emb_vect.transform(tokenized_sentences)
    assert(list(X_vectors_w2v[0]) == [0.5, 0.5])

def test_value_mean_embedding_vectorizer_with_partial_mapping_in_vocabulary():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    In this test function it's tested what's the result if we pass to the
    transform function a sentence composed by words with partial mapping
    in the vocabulary. The output should be a vector of length equal to
    the vector size, composed by the average of the vectors for the words
    that are present both in the sentence and in the vocabulary.

    Given:
    ======
    tokenized_sentences: list[list[str]]
                         The list contains the tokenized sentences (list of strings).
                         In this case it is used only a single sentence with partial
                         mapping between words from sentence and words in the 
                         vocabulary.

    dictionary_mapping_word_to_vec: dict
                                    Dictionary that maps each single word to a vector
                                    of dimension equal to 2.
    
    Tests:
    ======
            if the result is an array of dimension 2 (vector size is equal to two)
            composed by the average value of the vectors of the words present
            both in the vocabulary and in the sentence (in this case 'hello' 
            and 'world').
    '''
    tokenized_sentences = [['hello', 'world', 'i', 'am', 'flavio']]
    dictionary_mapping_word_to_vec = {'hello': [0, 1], 'world': [1, 0]}
    mean_emb_vect = MeanEmbeddingVectorizer(dictionary_mapping_word_to_vec)
    X_vectors_w2v = mean_emb_vect.transform(tokenized_sentences)
    assert(list(X_vectors_w2v[0]) == [0.5, 0.5])


def test_value_mean_embedding_vectorizer_multiple_sentences():
    '''
    Test function to test the behaviour of MeanEmbeddingVectorizer class.
    This test function tests that if we pass to the transform function two
    sentences, the output vectors that represent them are indipendent:
    we expect that the vector of the first sentence is made by the 
    average values for the words present only in the first sentence, while
    the vector of the second sentence is made by the average values for the
    words present in the second sentence.

    Given:
    ======
    tokenized_sentences: list[list[str]]
                         The list contains the tokenized sentences (list of strings).
                         Two different tokenized sentences.

    dictionary_mapping_word_to_vec: dict
                                    Dictionary that maps each single word to a vector
                                    of dimension equal to 3.
    
    Tests:
    ======
            if the two vector sentences are indipendent: the first number vector
            is composed by the average value of the vector words present in the first
            sentence, while the second vector sentence is composed by the average value 
            of the vector words present in the second sentence.
    '''
    tokenized_sentences = [['hello', 'world'], ['hello', 'world', 'i', 'am', 'flavio']]
    dictionary_mapping_word_to_vec = {'hello': [0, 0, 3], 'world': [0, 3, 0], 'flavio': [3, 0, 0]}
    mean_emb_vect = MeanEmbeddingVectorizer(dictionary_mapping_word_to_vec)
    X_vectors_w2v = mean_emb_vect.transform(tokenized_sentences)
    assert((X_vectors_w2v == np.array([[0, 1.5, 1.5], [1, 1, 1]])).all())

def test_vocabulary_include_special_chars_vector_layer():
    '''
    Test function to test the behaviour of init_vector_layer function.
    In this test function we test if the vocabulary that we pass to the
    function is correctly assigned to the vector layer with 
    special tokens ('' and  '[UNK]'); '[UNK]' stands for the unknown words).

    Given:
    ======
    vocabulary: list[str]
                List of tokens, that represents the vocabulary.
    
    Tests:
    ======
            if the initialized vector layer has the correct 
            vocabulary with special tokens (also '' and '[UNK]').
    '''
    vocabulary = ['hello', 'world']
    vector_layer = init_vector_layer(2, vocabulary)
    vocabulary_with_special_tokens = list(vector_layer.get_vocabulary(include_special_tokens = True))
    assert( vocabulary_with_special_tokens == ['', '[UNK]', 'hello', 'world'])

@settings(deadline = None)
@given(vector_length = st.integers(min_value = 0, max_value = 300))
def test_shape_vectorize_vector_layer_vector_length_variable(vector_length):
    '''
    Test function to test the behaviour of vectorize_X_data_tf function.
    In particular, this test function tests the output shape of vectors.
    We expect that the output shape is given by (number of sentences,
    vector length). In this test function the number of sentence is set to
    one (single sentence) and the vector length is decided by a strategy.

    @given:
    ========
    sentence: list[str]
              List containing a single sentence.

    vector_length: int
                   It's the vector size that we want to give to
                   the output vector. In this test case vector_length
                   is chosen by a strategy.

    Tests:
    =======
            if the output shape is equal to (1, vector_length), since
            we pass a single sentence and the embedding vector size 
            is given by vector_length.
    '''
    sentence = ['']
    vector_layer = init_vector_layer(vector_length,  uniq_vocabulary = ['hello'])
    vector = vectorize_X_data_tf(sentence, vector_layer)
    assert(vector.shape == (1, vector_length))


@settings(deadline = None)
@given(sentences = st.lists(st.text(), min_size = 1))
def test_shape_vectorize_vector_layer_variable_number_of_senteces(sentences):
    '''
    Test function to test the behaviour of vectorize_X_data_tf function.
    In particular, this test function tests the output shape of vectors.
    We expect that the output shape is given by (number of sentences,
    vector length). In this test function the vector legnth is set to
    1, while the sentences are decided by a strategy.

    @given:
    ========
    sentences: list[str]
               List containing a variable number sentences (at least one),
               generated by a strategy.

    vector_length: int
                   It's the vector size that we want to give to
                   the output vector. In this test case is set
                   equal to 1.

    Tests:
    =======
            if the output shape is equal to (number_of_sentence, 1), since
            we set the vector length equal to one and we pass a variable number
            of sentences.
    '''
    vector_length = 1
    vector_layer = init_vector_layer(vector_length, uniq_vocabulary = ['hello'])
    vector = vectorize_X_data_tf(sentences, vector_layer)
    assert(vector.shape == (len(sentences), 1))

def test_value_vectorization_all_words_in_vocabulary():
    '''
    Test function for vectorize_X_data_tf.
    In this test function it's tested what's the result if 
    we pass to the function a single sentence composed only by 
    words that are present in the vocabulary.
    We expect that the output vector is composed by the 
    integer index of the words in the vocabulary.
    The vocabulary will be something like this:
    index   word
    0       ''
    1       '[UNK]'
    2       'hello'
    3       'world'
    This means that we expect to have a result equal to [2,3].

    Given:
    ======
    vocabulary: list[str]
                List composed by the words that correspond to the
                vocabulary.
    
    sentences: list[str]
               Sentences that we want to vectorize. 
               In this test case all the words from the sentence 
               are also present in the vocabulary.
    
    Tests:
    ======
            if the output vector is equal to [2, 3], because the 
            words 'hello' and 'world' are respectively indexed 
            (in the layer vocabulary) with 2 and 3.
    '''
    vocabulary = ['hello', 'world']
    sentences = ['hello world']
    vector_layer = init_vector_layer(2, vocabulary)
    vector = vectorize_X_data_tf(sentences, vector_layer)
    assert(vector.numpy().tolist() == [[2, 3]])

def test_value_vectorization_empty_string():
    '''
    Test function for vectorize_X_data_tf.
    In this test function it's tested what's the result if 
    we pass to the function a single sentence composed by an
    empty character ('').
    The vocabulary will be something like this:
    index   word
    0       ''
    1       '[UNK]'
    2       'hello'
    3       'world'
    This means that we expect to have a result equal to [0].

    Given:
    ======
    vocabulary: list[str]
                List composed by the words that correspond to the
                vocabulary.
    
    sentences: list[str]
               Sentences that we want to vectorize. 
               In this test case we have a single sentence composed
               by an empty character ('').
    
    Tests:
    ======
            if the output vector is equal to [0].
    '''
    vocabulary = ['hello', 'world']
    sentences = ['']
    vector_layer = init_vector_layer(1, vocabulary)
    vector = vectorize_X_data_tf(sentences, vector_layer)
    assert(vector.numpy().tolist() == [[0]])

def test_value_vectorization_all_words_not_in_vocabulary():
    '''
    Test function for vectorize_X_data_tf.
    In this test function it's tested what's the result if 
    we pass to the function a single sentence composed only
    by words not present in the vocabulary.
    The vocabulary will be something like this:
    index   word
    0       ''
    1       '[UNK]'
    2       'hello'
    3       'world'
    This means that we expect to have a result equal to [1, 1],
    since the vector length is equal to 2 and the words are all
    equal to '[UNK]' (unknown).

    Given:
    ======
    vocabulary: list[str]
                List composed by the words that correspond to the
                vocabulary.
    
    sentences: list[str]
               Sentences that we want to vectorize. 
               In this test case we have a single sentence composed
               words not present in the vocabulary.
    
    vector_length: int
                   Vector size for the output vector.

    Tests:
    ======
            if the output vector is equal to [1, 1], since the vector length
            is equal to two and both the words are not present in the vocabulary.
    '''
    vocabulary = ['hello', 'world']
    sentences = ['hi flavio']
    vector_length = 2
    vector_layer = init_vector_layer(vector_length, vocabulary)
    vector = vectorize_X_data_tf(sentences, vector_layer)
    assert(vector.numpy().tolist() == [[1, 1]])

def test_truncation():
    '''
    Test function for vectorize_X_data_tf.
    In this test function it's tested the truncation
    operation. If we have a sentence longer than the 
    vector length value, we expect to truncate the 
    sentence.
    In this particular case we have a sentence of 3
    words, but the vector length is equal to 2. 
    We expect the output to be a vector that represents
    only the first two words.

    Given:
    ======
    vocabulary: list[str]
                List composed by the words that correspond to the
                vocabulary.
    
    sentences: list[str]
               Sentences that we want to vectorize. 
               In this test case we have a single sentence composed
               by three words.
    
    vector_length: int
                   Vector size for the output vector.
                   In this test case it is equal to two.

    Tests:
    ======
            if the final vector is composed by the integer indexes
            of the first two words, because of the truncation operation.
    '''
    vocabulary = ['hello', 'world']
    sentences = ['hello beautiful world']
    vector_length = 2
    vector_layer = init_vector_layer(vector_length, vocabulary)
    vector = vectorize_X_data_tf(sentences, vector_layer)
    # hello beautiful is vectorized in [2, 1]
    assert(vector.numpy().tolist() == [[2, 1]]) 

def test_padding():
    '''
    Test function for vectorize_X_data_tf.
    In this test function it's tested the padding
    operation. If we have a sentence shorter than the 
    vector length value, we expect to pad the sequence.
    In this particular case we have a sentence of a single
    word, but the vector length is equal to 5. 
    We expect the output vector equal to the index word, followed
    by four zeros (in this case we have [3, 0, 0, 0, 0], since 
    'world' is indexed with 3 in the layer vocabulary).

    Given:
    ======
    vocabulary: list[str]
                List composed by the words that correspond to the
                vocabulary.
    
    sentences: list[str]
               Sentences that we want to vectorize. 
               In this test case we have a single sentence composed
               by a single word  ('world').
    
    vector_length: int
                   Vector size for the output vector.
                   In this test case it is equal to five.

    Tests:
    ======
            if the final vector is equal to the index of the word in
            the vocabulary, followed by four zeros. 
            In this case we will have [3, 0, 0, 0, 0], because 
            the 'world' token is indexed with 3 in the vocabulary.
    '''
    vocabulary = ['hello', 'world']
    sentences = ['world']
    vector_length = 5
    vector_layer = init_vector_layer(vector_length, vocabulary)
    vector = vectorize_X_data_tf(sentences, vector_layer)
    assert(vector.numpy().tolist() == [[3, 0, 0, 0, 0]])

def test_value_vectorization_partial_mapping():
    '''
    Test function for vectorize_X_data_tf.
    In this test function it's tested what's the result
    if there isn't a perfect match between words from the sentence
    and words from the vocabulary.
    We expect to have 1 in the vector output, if the word is unknown
    (it mean's that is not present in the vocabulary).
    We expect to have an integer number greater than 1 for the words
    present in the vocabulary.

    Given:
    ======
    vocabulary: list[str]
                List composed by the words that correspond to the
                vocabulary.
    
    sentences: list[str]
               Sentences that we want to vectorize. 
               In this test case we have a single sentence composed
               by five words, that some of them aren't part of the 
               vocabulary.
    
    vector_length: int
                   Vector size for the output vector.
                   In this test case it is equal to five.

    Tests:
    ======
            if the final vector is equal to a 5 long vector of integer, 
            with 1 if words are unknown (not present in vocabulary) or
            an integer number greater than 1 if words are part of the vocabulary
            (the integer number corresponds to the index number inside the 
            layer vocabulary).
    '''
    vocabulary = ['hello', 'world', 'flavio']
    sentences = ['hello world i am flavio']
    vector_length = 5
    vector_layer = init_vector_layer(vector_length, vocabulary)
    vector = vectorize_X_data_tf(sentences, vector_layer)
    assert(vector.numpy().tolist() == [[2, 3, 1, 1, 4]])

def test_value_vectorization_layer_two_sentences():
    '''
    Test function for vectorize_X_data_tf.
    In this test function it's tested what's the 
    value result if two sentences are passed to the
    function.
    The first sentence is 2 words long and the 
    vector length is equel to three: we expect
    to pad the sequence.
    The second sentence is 3 words long, but the 
    first two words are not present in the vocabulary:
    we expect to have two 1 values in the second vector.

    Given:
    ======
    vocabulary: list[str]
                List composed by the words that correspond to the
                vocabulary.
    
    sentences: list[str]
               Sentences that we want to vectorize. 
               In this test case we use two sentences: the first
               one must be padded (since it is 2 words long and the
               vector length is equal to 3), while in the second
               sequence will contain some 1, because in the second
               sentence there are some unknown words.
    
    vector_length: int
                   Vector size for the output vector.
                   In this test case it is equal to three.

    Tests:
    ======
           if the output vector is equal to [[2, 3,0], [1, 1, 4]]:
           since the first sentence ([2, 3, 0]) is two words long and
           it must be padded with a final zero, while the second 
           sentence ([1, 1, 4]) contains two unknown words, that are
           vectorized with 1.
    '''
    vocabulary = ['hello', 'world', 'flavio']
    sentences = ['hello world', 'i am flavio']
    vector_length = 3
    vector_layer = init_vector_layer(vector_length, vocabulary)
    vector = vectorize_X_data_tf(sentences, vector_layer)
    print(vector)
    assert(vector.numpy().tolist() == [[2, 3, 0],  [1, 1, 4]])
