'''
Test functions for the reading and writing functions.
'''
from text_analysis.read_write_data import handle_multiple_occurencies
from text_analysis.read_write_data import get_paths, remove_none_from_dict

def test_train_name():
    '''
    Test function for the function get_paths.
    In this test function it is tested that
    file name for 'train' is correctly matched.

    Given:
    ======
    filenames: list[str]
               Sequence of possible file names.

    Tests:
    ======
            If the match between the dictionary key
            'train' and the file name is correct.
    '''
    filenames = ['random_train.csv',
                 'random_val.csv',
                 'random_test.csv']
    dict_file_name = get_paths(filenames)
    assert(dict_file_name['train'] == 'random_train.csv')

def test_val_name():
    '''
    Test function for the function get_paths.
    In this test function it is tested that
    file name for 'val' is correctly matched.

    Given:
    ======
    filenames: list[str]
               Sequence of possible file names.

    Tests:
    ======
            If the match between the dictionary key
            'val' and the file name is correct.
    '''
    filenames = ['random_train.csv',
                 'random_val.csv',
                 'random_test.csv']
    dict_file_name = get_paths(filenames)
    assert(dict_file_name['val'] == 'random_val.csv')

def test_test_name():
    '''
    Test function for the function get_paths.
    In this test function it is tested that
    file name for 'test' is correctly matched.

    Given:
    ======
    filenames: list[str]
               Sequence of possible file names.

    Tests:
    ======
            If the match between the dictionary key
            'test' and the file name is correct.
    '''
    filenames = ['random_train.csv',
                 'random_val.csv',
                 'random_test.csv']
    dict_file_name = get_paths(filenames)
    assert(dict_file_name['test'] == 'random_test.csv')

def test_no_match():
    '''
    Test function for the function get_paths.
    In this test function it is tested what's 
    happening when there is no match between
    file names and "special" words ('train', 'test',
    'val').

    Given:
    ======
    filenames: list[str]
               File names for csv files 
               containing only one single name
               without "special" words.

    Tests:
    ======
            If there is no match between the dictionary 
            keys and the file name, it is 
            returned only an empty list.
    '''
    filenames = ['random.csv']
    dict_file_name = get_paths(filenames)
    assert(dict_file_name['train'] == [])

def test_get_paths_len_dict():
    '''
    Test function for the function get_paths.
    In this test function it is tested that there
    are always three keys inside the dictionary,
    also if there is no match at all.

    Given:
    ======
    filenames: list[str]
               File names for csv files 
               containing only one single name
               without "special" words.

    Tests:
    ======
            If there is no match between the dictionary 
            keys and the file name, the dictionary keys
            are always three.
    '''
    filenames = ['random.csv']
    dict_file_name = get_paths(filenames)
    assert(len(dict_file_name) == 3)

def test_get_paths_len_dict():
    '''
    Test function for the function get_paths.
    In this test function it is tested that there
    are always these keys: 'train', 'test', 'val'.

    Given:
    ======
    filenames: list[str]
               File names for csv files 
               containing only one single name
               without "special" words.

    Tests:
    ======
            If there is no match between the dictionary 
            keys and the file name, the dictionary keys
            are always: 'train', 'val', 'test'.
    '''
    filenames = ['random.csv']
    dict_file_name = get_paths(filenames)
    assert(list(dict_file_name.keys()) == ['train', 'val', 'test'])

def test_train_name_twofilenames():
    '''
    Test function for the function get_paths.
    In this test function it is tested what's
    the result if a list of two file names is passed.
    In one file name there is 'train' string, while in the 
    other there isn't a special word. The function 
    matches the second name as test file.

    Given:
    ======
    filenames: list[str]
               Sequence of two file names: one for
               train ('train' word), the other without
               special word.

    Tests:
    ======
            if the 'train' file is correctly matched.
    '''
    filenames = ['random_train.csv',
                 'random.csv']
    dict_file_name=get_paths(filenames)
    assert(dict_file_name['train'] == 'random_train.csv')

def test_test_name_twofilenames():
    '''
    Test function for the function get_paths.
    In this test function it is tested what's
    the result if a list of two file names is passed.
    In one file name there is 'train' string, while in the 
    other there isn't a special word. The function 
    matches the second name as test file.

    Given:
    ======
    filenames: list[str]
               Sequence of two file names: one for
               train ('train' word), the other without
               special word.

    Tests:
    ======
            if the 'test' file is correctly matched with the 
            file that has not special word in its name.
    '''
    filenames = ['random_train.csv',
                 'random.csv']
    dict_file_name=get_paths(filenames)
    assert(dict_file_name['test'] == 'random.csv')

def test_train_name_capital_letter():
    '''
    Test function for the function get_paths.
    In this test function it's tested what's 
    the result if inside the file name there
    are capital letters in 'train'.

    Given:
    ======
    filenames: list[str]
               Sequence of file names.
               The train file name contains some
               capital letters within 'train' string.
    Tests:
    ======
            if there is a good match between the dictionary
            key 'train' and the file name given. 
    '''
    filenames = ['random_TrAiN.csv',
                 'random_val.csv',
                 'random_test.csv']
    dict_file_name = get_paths(filenames)
    assert(dict_file_name['train'] == 'random_TrAiN.csv')

def test_val_name_capital_letter():
    '''
    Test function for the function get_paths.
    In this test function it's tested what's 
    the result if inside the file name there
    are capital letters in 'val'.

    Given:
    ======
    filenames: list[str]
               Sequence of file names.
               The validation file name contains some
               capital letters within 'val' string.

    Tests:
    ======
            if there is a good match between the dictionary
            key 'val' and the file name given. 
    '''
    filenames = ['random_train.csv',
                 'random_VaL.csv',
                 'random_test.csv']
    dict_file_name = get_paths(filenames)
    assert(dict_file_name['val'] == 'random_VaL.csv')

def test_test_name_capital_letter():
    '''
    Test function for the function get_paths.
    In this test function it's tested what's 
    the result if inside the file name there
    are capital letters in 'test'.

    Given:
    ======
    filenames: list[str]
               Sequence of file names.
               The test file name contains some
               capital letters within 'test' string.

    Tests:
    ======
            if there is a good match between the dictionary
            key 'test' and the file name given. 
    '''
    filenames = ['random_train.csv',
                 'random_val.csv',
                 'random_TeST.csv']
    dict_file_name = get_paths(filenames)
    assert(dict_file_name['test'] == 'random_TeST.csv')

def test_remove_emptylist_from_dict():
    '''
    Test function for remove_none_from_dict.
    In this test function it is tested what's the result 
    if the input dictionary contain an empty list ([]).

    Given:
    =======
    dict_with_empty_list: dict
                          Dictionary with empty a list.

    Tests:
    =======
            The output dictionary should be empty.
    '''
    dict_with_empty_list = {'empty list': []}
    dict_no_empty_list = remove_none_from_dict(dict_with_empty_list)
    assert(dict_no_empty_list == {})

def test_remove_emptylist_from_dict_with_other_keys():
    '''
    Test function for remove_none_from_dict.
    In this test function it is tested what's the result 
    if the input dictionary contain an empty list ([]) and
    other keys mapped not to empty list.

    Given:
    =======
    dict_with_empty_list: dict
                          Dictionary with a key mapped to an empty list
                          and other keys mapped not to empty list.

    Tests:
    =======
            The output dictionary should be composed only by keys mapped 
            not to empty list.
    '''
    dict_with_empty_list = {'empty list': [], 'other': 1, 'a': 'a'}
    dict_no_empty_list = remove_none_from_dict(dict_with_empty_list)
    print(dict_no_empty_list)
    assert(dict_no_empty_list == {'other': 1, 'a': 'a'})

def test_remove_none_from_dict():
    '''
    Test function for remove_none_from_dict.
    In this test function it is tested what's the result 
    if the input dictionary contain a key mapped to None.

    Given:
    =======
    dict_with_None: dict
                    Dictionary with a key mapped to a None value.

    Tests:
    =======
            The output dictionary should be an empty dictionary.
    '''
    dict_with_empty_list = {'none': None}
    dict_no_empty_list = remove_none_from_dict(dict_with_empty_list)
    assert(dict_no_empty_list == {})

def test_remove_zeros_from_dict():
    '''
    Test function for remove_none_from_dict.
    In this test function it is tested what's the result 
    if the input dictionary contain a key mapped to 0.

    Given:
    =======
    dict_with_None: dict
                    Dictionary with a key mapped to a 0 value.

    Tests:
    =======
            The output dictionary should be an empty dictionary.
    '''
    dict_with_empty_list = {'zero': 0}
    dict_no_empty_list = remove_none_from_dict(dict_with_empty_list)
    assert(dict_no_empty_list == {})

def test_train_multiple_occurencies():
    '''
    Test function for handle_multiple_occurencies.
    In this test function it's tested what's the result
    if there are multiple matches for the 'train' word.

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'train' word within the string.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'train' within the string.
    '''
    multiple_matches = ['constrain_train',
                        'constrain_val',
                        'constrain_test']
    train_name = handle_multiple_occurencies(multiple_matches, 'train')
    assert(train_name == 'constrain_train')

def test_val_multiple_occurencies():
    '''
    Test function for handle_multiple_occurencies.
    In this test function it's tested what's the result
    if there are multiple matches for the 'val' word.

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'val' word within the string.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'val' within the string.
    '''
    multiple_matches = ['eigenvalue_train',
                        'eigenvalue_val',
                        'eigenvalue_test']
    val_name = handle_multiple_occurencies(multiple_matches, 'val')
    assert(val_name == 'eigenvalue_val')

def test_val_multiple_occurencies():
    '''
    Test function for handle_multiple_occurencies.
    In this test function it's tested what's the result
    if there are multiple matches for the 'test' word.

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'test' word within the string.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'test' within the string.
    '''
    multiple_matches = ['protest_train',
                        'protest_val',
                        'protest_test']
    test_name = handle_multiple_occurencies(multiple_matches, 'test')
    assert(test_name == 'protest_test')

def test_train_multiple_occurencies_cap_letters():
    '''
    Test function for handle_multiple_occurencies.
    In this test function it's tested what's the result
    if there are multiple matches for the 'train' word 
    considering capital letters.

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'train' word within the string 
                      with some capital letters.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'train' within the string.
    '''
    multiple_matches = ['consTrain_train',
                        'consTrain_val',
                        'consTrain_test']
    train_name = handle_multiple_occurencies(multiple_matches, 'train')
    assert(train_name == 'consTrain_train')


def test_val_multiple_occurencies_cap_letters():
    '''
    Test function for handle_multiple_occurencies.
    In this test function it's tested what's the result
    if there are multiple matches for the 'val' word 
    considering capital letters.

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'val' word within the string 
                      with some capital letters.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'val' within the string.
    '''
    multiple_matches = ['eigenValue_train',
                        'eigenValue_val',
                        'eigenValue_test']
    val_name = handle_multiple_occurencies(multiple_matches, 'val')
    assert(val_name == 'eigenValue_val')

def test_val_multiple_occurencies_cap_letters():
    '''
    Test function for handle_multiple_occurencies.
    In this test function it's tested what's the result
    if there are multiple matches for the 'test' word 
    considering capital letters.

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'test' word within the string 
                      with some capital letters.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'test' within the string.
    '''
    multiple_matches = ['proTest_train',
                        'proTest_val',
                        'proTest_test']
    test_name = handle_multiple_occurencies(multiple_matches, 'test')
    assert(test_name == 'proTest_test')