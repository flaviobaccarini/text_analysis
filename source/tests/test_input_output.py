'''
Test functions for the reading and writing functions.
'''
from text_analysis.read_write_data import handle_multiple_occurencies
from text_analysis.read_write_data import get_filenames, remove_none_from_dict

def test_train_name():
    '''
    Test function for the function get_filenames.
    In this test function it is tested that
    file name for 'train' is correctly matched
    with the value for the key 'train' in the
    output dictionary from the function.

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
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['train'] == 'random_train.csv')

def test_val_name():
    '''
    Test function for the function get_filenames.
    In this test function it is tested that
    file name for 'val' is correctly matched
    with the value for the key 'val' in the
    output dictionary from the function.

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
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['val'] == 'random_val.csv')

def test_test_name():
    '''
    Test function for the function get_filenames.
    In this test function it is tested that
    file name for 'test' is correctly matched
    with the value for the key 'test' in the
    output dictionary from the function.

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
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['test'] == 'random_test.csv')

def test_no_match_train():
    '''
    Test function for the function get_filenames.
    In this test function it is tested what's 
    happening when there is no match between
    file names and "special" words ('train', 'test',
    'val').

    Given:
    ======
    filenames: list[str]
               File names for csv files 
               containing names
               without "special" words.

    Tests:
    ======
            If there is no match between the dictionary 
            keys and the input file names, each key is 
            mapped to an empty list (in this test function
            it is tested the case with 'train' key).
    '''
    filenames = ['random.csv',
                 'random1.csv',
                 'random2.csv']
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['train'] == [])

def test_no_match_val():
    '''
    Test function for the function get_filenames.
    In this test function it is tested what's 
    happening when there is no match between
    file names and "special" words ('train', 'test',
    'val').

    Given:
    ======
    filenames: list[str]
               File names for csv files 
               containing names
               without "special" words.

    Tests:
    ======
            If there is no match between the dictionary 
            keys and the input file names, each key is 
            mapped to an empty list (in this test function
            it is tested the case with 'val' key).
    '''
    filenames = ['random.csv',
                 'random1.csv',
                 'random2.csv']
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['val'] == [])

def test_no_match_test():
    '''
    Test function for the function get_filenames.
    In this test function it is tested what's 
    happening when there is no match between
    file names and "special" words ('train', 'test',
    'val').

    Given:
    ======
    filenames: list[str]
               File names for csv files 
               containing names
               without "special" words.

    Tests:
    ======
            If there is no match between the dictionary 
            keys and the input file names, each key is 
            mapped to an empty list (in this test function
            it is tested the case with 'test' key).
    '''
    filenames = ['random.csv',
                 'random1.csv',
                 'random2.csv']
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['test'] == [])

def test_train_name_twofilenames():
    '''
    Test function for the function get_filenames.
    In this test function it is tested what's
    the result if a list of two file names is passed.
    In one file name there is 'train' string, while in the 
    other there isn't a special word. The function 
    matches the first file as train file.

    Given:
    ======
    filenames: list[str]
               Sequence of two file names: one for
               train ('train' word), the other without
               special word.

    Tests:
    ======
            if the 'train' key in dictionary
            is correctly matched with the correct
            'train' file.
    '''
    filenames = ['random_train.csv',
                 'random.csv']
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['train'] == 'random_train.csv')

def test_test_name_twofilenames():
    '''
    Test function for the function get_filenames.
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
            if the 'test' key is correctly matched with the 
            file that has not special word in its name.
    '''
    filenames = ['random_train.csv',
                 'random.csv']
    dict_file_name=get_filenames(filenames)
    assert(dict_file_name['test'] == 'random.csv')

def test_train_name_capital_letter():
    '''
    Test function for the function get_filenames.
    In this test function  it's tested if the
    get_filenames function is indipendent from the
    presence of capital letters inside "special" 
    words ('train', 'val' and 'test').
    Case for 'train' key.

    Given:
    ======
    filenames: list[str]
               Sequence of file names.
               The train file name contains some
               capital letters within 'train' string.
    Tests:
    ======
            if there is a good match between the dictionary
            key 'train' and the file name given, despite of
            capital letters presence.
    '''
    filenames = ['random_TrAiN.csv',
                 'random_val.csv',
                 'random_test.csv']
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['train'] == 'random_TrAiN.csv')

def test_val_name_capital_letter():
    '''
    Test function for the function get_filenames.
    In this test function  it's tested if the
    get_filenames function is indipendent from the
    presence of capital letters inside "special" 
    words ('train', 'val' and 'test').
    Case for 'val' key.

    Given:
    ======
    filenames: list[str]
               Sequence of file names.
               The validation file name contains some
               capital letters within 'val' string.

    Tests:
    ======
            if there is a good match between the dictionary
            key 'val' and the file name given, despite of
            capital letters presence.
    '''
    filenames = ['random_train.csv',
                 'random_VaL.csv',
                 'random_test.csv']
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['val'] == 'random_VaL.csv')

def test_test_name_capital_letter():
    '''
    Test function for the function get_filenames.
    In this test function  it's tested if the
    get_filenames function is indipendent from the
    presence of capital letters inside "special" 
    words ('train', 'val' and 'test').
    Case for 'test' key.

    Given:
    ======
    filenames: list[str]
               Sequence of file names.
               The test file name contains some
               capital letters within 'test' string.

    Tests:
    ======
            if there is a good match between the dictionary
            key 'test' and the file name given, despite of 
            capital letters presence.
    '''
    filenames = ['random_train.csv',
                 'random_val.csv',
                 'random_TeST.csv']
    dict_file_name = get_filenames(filenames)
    assert(dict_file_name['test'] == 'random_TeST.csv')

def test_remove_emptylist_from_dict():
    '''
    Test function for remove_none_from_dict.
    In this test function it is tested what's the result 
    if the input dictionary contain an empty list ([]).

    Given:
    =======
    dict_with_empty_list: dict
                          Dictionary with an empty list.

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

def test_train_multiple_occurencies():
    '''
    Test function for handle_multiple_occurencies.
    In this test function it's tested what's the result
    if there are multiple matches for the 'train' word.
    The output should be the string that contains the 
    greatest number of 'train' within the text.

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of strings that contain at least
                      one time the 'train' word within the string.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'train' within the text.
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
    The output should be the string that contains the 
    greatest number of 'val' within the text.

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'val' word within the string.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'val' within the text.
    '''
    multiple_matches = ['eigenvalue_train',
                        'eigenvalue_val',
                        'eigenvalue_test']
    val_name = handle_multiple_occurencies(multiple_matches, 'val')
    assert(val_name == 'eigenvalue_val')

def test_test_multiple_occurencies():
    '''
    Test function for handle_multiple_occurencies.
    In this test function it's tested what's the result
    if there are multiple matches for the 'test' word.
    The output should be the string that contains the 
    greatest number of 'test' within the text.

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'test' word within the string.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'test' within the text.
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
    with capital letters (indipendence of the 
    handle_multiple_occurencies from the presence of capital
    letters within 'train').

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'train' word within the string 
                      with capital letters.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'train' (no matter if lowercase or 
            capital letters) within the string.
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
    with capital letters (indipendence of the 
    handle_multiple_occurencies from the presence of capital
    letters within 'val').

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'val' word within the string 
                      with capital letters.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'val' (no matter if lowercase or 
            capital letters) within the string.
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
    with capital letters (indipendence of the 
    handle_multiple_occurencies from the presence of capital
    letters within 'test').

    Given:
    ======
    multiple_matches: list[str]
                      Sequence of string that contains at least
                      one time the 'test' word within the string 
                      with capital letters.
    
    Tests:
    ======
            The output of the function should be the string with the
            maximum number of 'test' (no matter if lowercase or 
            capital letters) within the string.
    '''
    multiple_matches = ['proTest_train',
                        'proTest_val',
                        'proTest_test']
    test_name = handle_multiple_occurencies(multiple_matches, 'test')
    assert(test_name == 'proTest_test')