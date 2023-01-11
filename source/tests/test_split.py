'''
Test functions for the splitting module.
'''
from hypothesis.extra.pandas import column, data_frames
from text_analysis.split_dataframe import split_single_dataframe
from text_analysis.split_dataframe import split_two_dataframes
import pandas as pd
from hypothesis import strategies as st
from hypothesis import given
import unittest

@given(singledataframe = data_frames([column('col', dtype = str)]))
def test_split_df_single_dataset(singledataframe):
    '''
    Test function to test the behaviour of split_single_dataframe function.
    In this test function is tested that the function returns three elements
    starting from a generic pandas DataFrame.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset.
    
    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the output of the function is composed by three elements
            (train, validation, test dataframe).
    '''
    # initialize the train and test fraction
    train_frac, test_frac = 0.80, 0.10
    # split
    df_split = split_single_dataframe(singledataframe, (train_frac, test_frac), seed = 42)
    # verify that df_split is composed by three elements:
    assert(len(df_split) == 3) 


@given(singledataframe = data_frames([column('col', dtype = str)]))
def test_split_df_single_dataset_dimension_train_df(singledataframe):
    '''
    Test function to test the behaviour of split_single_dataframe function.
    In this test function is tested that the df_train has the correct 
    dimension given by the train fraction number.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the df_train (train dataframe) has the correct dimension
            (given by the train fraction).
    '''
    # initialize the train and test fraction
    train_frac, test_frac = 0.80, 0.10
    # split
    df_train, df_val, df_test = split_single_dataframe(singledataframe, (train_frac, test_frac), seed = 42)
    # verify that train dataframe has the correct dimensionality.    
    assert(len(df_train) == 
                round(train_frac * (len(df_train) + len(df_val) + len(df_test))))

@given(singledataframe = data_frames([column('col', dtype = str)]))
def test_split_df_single_dataset_dimension_test_df(singledataframe):
    '''
    Test function to test the behaviour of split_single_dataframe function.
    In this test function is tested that the df_test has the correct 
    dimension given by the test fraction number.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the df_test (test dataframe) has the correct dimension
            (given by the test fraction).
    '''
    # initialize the train and test fraction
    train_frac, test_frac = 0.80, 0.10
    # split
    df_train, df_val, df_test = split_single_dataframe(singledataframe, (train_frac, test_frac), seed = 42)
    # verify that test dataframe has the correct dimensionality.    
    assert(len(df_test) == 
                round(test_frac * (len(df_train) + len(df_val) + len(df_test))))

@given(singledataframe = data_frames([column('col', dtype = str)]))
def test_split_df_single_dataset_dimension_val_df(singledataframe):
    '''
    Test function to test the behaviour of split_single_dataframe function.
    In this test function is tested that the df_val has the correct 
    dimension. The dimension is given by the length of the original dataframe
    minus the length of the train and test dataframe.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the df_val (validation dataframe) has the correct dimension:
            which is the length of the initial dataframe minus the length
            of the train and test dataframe.
    '''
    # initialize the train and test fraction
    train_frac, test_frac = 0.80, 0.10
    # split
    df_train, df_val, df_test = split_single_dataframe(singledataframe, (train_frac, test_frac), seed = 42)
    # verify that validation dataframe has the correct dimensionality.    
    assert(len(df_val) == 
                len(singledataframe) - len(df_train) - len(df_test))

def test_intersection_split_singldf_train_test():
    '''
    Test function to test the behaviour of split_single_dataframe.
    In particular, in this test function the empty intersection between split dataframes
    is checked.
    We expect that the intersection between train and test dataframe is empty.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the intersection between train and test dataframe is empty.
    '''
    singledataframe = pd.DataFrame({'label': ['A', 'A', 'A', 'B', 'B'],
                                    'text': ['random', 'try', 'test', 'well', 'good']})
    train_fract, test_fract = 0.80, 0.10
    df_train, _, df_test = split_single_dataframe(singledataframe, (train_fract, test_fract), seed = 42)
    intersection_train_test = pd.merge(df_train, df_test, how = 'inner')
    assert(intersection_train_test.empty)

def test_intersection_split_singldf_train_val():
    '''
    Test function to test the behaviour of split_single_dataframe.
    In particular, in this test function the empty intersection between split dataframes
    is checked. 
    We expect that the intersection between train and validation dataframe is empty.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the intersection between train and validation dataframe is empty.
    '''
    singledataframe = pd.DataFrame({'label': ['A', 'A', 'A', 'B', 'B'],
                                    'text': ['random', 'try', 'test', 'well', 'good']})
    train_fract, test_fract = 0.80, 0.10
    df_train, df_valid, _ = split_single_dataframe(singledataframe, (train_fract, test_fract), seed = 42)
    intersection_train_val = pd.merge(df_train, df_valid, how = 'inner')
    assert(intersection_train_val.empty)

def test_intersection_split_singldf_test_val():
    '''
    Test function to test the behaviour of split_single_dataframe.
    In particular, in this test function the empty intersection between split dataframes
    is checked. We expect that the intersection between test and validation dataframe is empty.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the intersection between validation and test dataframe is empty.
    '''
    singledataframe = pd.DataFrame({'label': ['A', 'A', 'A', 'B', 'B'],
                                    'text': ['random', 'try', 'test', 'well', 'good']})
    train_fract, test_fract = 0.80, 0.10
    _, df_valid, df_test = split_single_dataframe(singledataframe, (train_fract, test_fract), seed = 42)
    intersection_test_val = pd.merge(df_test, df_valid, how = 'inner')
    assert(intersection_test_val.empty)

def test_union_train_val_test_equal_singledf():
    '''
    Test function to test the behaviour of split_single_dataframe.
    In this test function it's tested that the union of all the split
    dataframes are equal to the first original dataframe.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the union between the split dataframes (train, validation, test) is equal
            to the original dataframe.
    '''
    singledataframe = pd.DataFrame({'label': ['A', 'A', 'A', 'B', 'B'],
                                    'text': ['random', 'try', 'test', 'well', 'good']})
    train_fract, test_fract = 0.80, 0.10
    df_train, df_valid, df_test = split_single_dataframe(singledataframe, (train_fract, test_fract), seed = 42)
    df_all = pd.merge(pd.merge(df_train, df_valid, how = 'outer'), df_test, how = 'outer')
    # resort the value: both for union and the original dataframe
    df_all = df_all.sort_values(by=df_all.columns.tolist()).reset_index(drop=True)
    singledataframe = singledataframe.sort_values(by=singledataframe.columns.tolist()).reset_index(drop=True)
    assert(df_all.equals(singledataframe))

@given(train_fract = st.floats(min_value = 0.01, max_value = 0.99))
def test_dimension_traindf_from_singledf_with_different_fract(train_fract):
    '''
    Test function for split_single_dataframe. 
    In this test function it's tested if the dimension of df_train
    (train dataframe) is correct testing different values for the 
    train fraction.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset generated by 
                           a strategy.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the df_train (train dataframe) has the correct dimension
            given by train fraction changing every time the fraction value.
    '''
    singledataframe = pd.DataFrame({'text': ['a', 'b', 'c', 'd', 'e',
                                             'f', 'g', 'h', 'i', 'l']})
    test_fract = 1 - train_fract
    df_train, df_val, df_test = split_single_dataframe(singledataframe, (train_fract, test_fract), seed = 42)                                                
    assert(len(df_train) == 
                round(train_fract * (len(df_train) + len(df_val) + len(df_test))))


@given(test_fract = st.floats(min_value = 0.01, max_value = 0.99))
def test_dimension_testdf_from_singledf_with_different_fract(test_fract):
    '''
    Test function for split_single_dataframe. 
    In this test function it's tested if the dimension of df_test
    (test dataframe) is correct testing different values for the test
    fraction.

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset generated by 
                           a strategy.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the df_test (test dataframe) has the correct dimension given
            by the test fraction with different fraction values.
    '''
    singledataframe = pd.DataFrame({'text': ['a', 'b', 'c', 'd', 'e',
                                             'f', 'g', 'h', 'i', 'l']})
    train_fract = 1 - test_fract
    df_train, df_val, df_test = split_single_dataframe(singledataframe, (train_fract, test_fract), seed = 42)                                                
    assert(len(df_test) == 
                round(test_fract * (len(df_train) + len(df_val) + len(df_test))))

@given(train_fract = st.floats(min_value = 0.01, max_value = 0.49),
       test_fract = st.floats(min_value = 0.01, max_value = 0.49))
def test_dimension_valdf_from_singledf_with_different_fract(train_fract, test_fract):
    '''
    Test function for split_single_dataframe. 
    In this test function it's tested if the dimension of df_val
    (validation dataframe) is correct if we use different values for
    train and test fraction (the train and test fractions can vary
    from 0.01 to 0.49).

    Given:
    =======
    train_frac, test_frac: floats
                           Float numbers that represent the fractions in which
                           the user wants to split the dataset generated by 
                           a strategy.

    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if the df_val (validation dataframe) has the correct dimension 
            with different fraction values. The dimension of df_val
            is equal to the length of the original dataframe minus the
            length of the train and test dataframe.
    '''
    singledataframe = pd.DataFrame({'text': ['a', 'b', 'c', 'd', 'e',
                                             'f', 'g', 'h', 'i', 'l']})
    df_train, df_val, df_test = split_single_dataframe(singledataframe, (train_fract, test_fract), seed = 42)                                                
    assert(len(df_val) == 
                len(singledataframe) - (len(df_train) + len(df_test)))

def test_negative_fraction_singledf():
    '''
    Test function for split_single_dataframe. 
    In this test function it's tested what's the result when 
    to the function is passed a negative number as an input fraction.

    Given:
    ======
    negative_fract: negative number
                    Negative float number that will pass as
                    fraction to the function.
    
    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if a negative fraction is passed to the function we expect
            a ValueError is raised.
    '''
    negative_fract = -0.2
    singledataframe = pd.DataFrame({'text': ['a', 'b', 'c', 'd', 'e',
                                             'f', 'g', 'h', 'i', 'l']})
    with unittest.TestCase.assertRaises(unittest.TestCase,
                                        expected_exception = ValueError):
        dfs_error = split_single_dataframe(singledataframe, (negative_fract, 0.2), seed = 42)

def test_fraction_greater_than_one_singledf():
    '''
    Test function for split_single_dataframe. 
    In this test function it's tested what's the result when 
    to the function is passed a fraction number greater than one
    as an input fraction.

    Given:
    ======
    fract_greater_than_one: number greater than 1
                            Number greater than one that will pass as
                            input fraction to the function.
    
    singledataframe: pd.DataFrame
                     The single dataset that will be split in three dataframes.
    
    Tests:
    ======
            if a fraction greater than one is passed to the 
            function we expect that a ValueError is raised.
    '''
    fract_greater_than_one = 1.2
    singledataframe = pd.DataFrame({'text': ['a', 'b', 'c', 'd', 'e',
                                             'f', 'g', 'h', 'i', 'l']})
    with unittest.TestCase.assertRaises(unittest.TestCase,
                                        expected_exception = ValueError):
        dfs_error = split_single_dataframe(singledataframe, (fract_greater_than_one, 0.2), seed = 42)


@given(df_train_valid = data_frames([column('col', dtype = str)]),
       df_only_test = data_frames([column('col', dtype = str)]))
def test_split_two_datasets_in_three(df_train_valid, df_only_test):
    '''
    Test function for split_two_dataframes.
    In this test function it's tested that the output of the function
    is composed by three elements starting from two generic dataframes.

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the function output is a sequence of three elements 
            (three split dataframes).
    '''
    # initialize the train and test fraction
    train_frac = 0.80
    # split
    df_split = split_two_dataframes([df_train_valid, df_only_test], train_frac, seed = 42)
    # verify that df_split is composed by three elements:
    assert(len(df_split) == 3) 


@given(df_train_valid = data_frames([column('col', dtype = str)]),
       df_original_test = data_frames([column('col', dtype = str)]))
def test_dimension_train_df_splitting_two_datasets(df_train_valid, df_original_test):
    '''
    Test function for split_two_dataframes.
    In this test function it's tested that the train dataframe
    df_train has the correct dimension given by the train fraction.

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the df_train has the correct dimension given by the train fraction.  
    '''
    # initialize the train and test fraction
    train_frac = 0.80
    # split
    df_train, _, _ = split_two_dataframes([df_train_valid, df_original_test], train_frac, seed = 42)
    # verify that df_train has the correct dimensionality
    assert(len(df_train) == round(train_frac*len(df_train_valid)))

@given(df_train_valid = data_frames([column('col', dtype = str)]),
       df_original_test = data_frames([column('col', dtype = str)]))
def test_testdf_splitting_two_datasets(df_train_valid, df_original_test):
    '''
    Test function for split_two_dataframes.
    In this test function it's tested that the test dataframe
    df_test is equal to the second dataset passed as parameter 
    to the function (in this case: df_only_test).

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the df_test is equal to the original test dataframe df_original_test.  
    '''
    # initialize the train and test fraction
    train_frac = 0.80
    # split
    _, _, df_test = split_two_dataframes([df_train_valid, df_original_test], train_frac, seed = 42)
    # verify that df_test is equal to df_original_test 
    assert(df_test.equals(df_original_test))


@given(df_train_valid = data_frames([column('col', dtype = str)]),
       df_original_test = data_frames([column('col', dtype = str)]))
def test_dimension_valdf_splitting_two_datasets(df_train_valid, df_original_test):
    '''  
    Test function for split_two_dataframes.
    In this test function it's tested that the validation dataframe
    df_val has the correct dimension, which is the length of the 
    first original dataframe minus the length of the train dataframe (df_train).

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the df_val has the correct dimension (length of df_train_valid
            minus the length of train dataframe).  
    '''
    # initialize the train and test fraction
    train_frac = 0.80
    # split
    df_train, df_val, _ = split_two_dataframes([df_train_valid, df_original_test], train_frac, seed = 42)
    # verify that df_val has the correct dimension 
    assert(len(df_val) == len(df_train_valid) - len(df_train))

def test_intersection_split_twodfs_train_test():
    '''
    Test function for split_two_dataframes.
    In this test function it's tested that the intersection, 
    after the splitting, between the train and test dataframe is empty.

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the intersection between df_train and df_test is empty.
    '''
    df_original_trainval = pd.DataFrame({'label': ['A', 'A', 'B'],
                                    'text': ['random', 'try', 'test']})
    df_original_test = pd.DataFrame({'label': ['B', 'A', 'B'],
                              'text': ['good', 'well', 'great']})
    train_fract = 0.50
    df_train, _, df_test = split_two_dataframes([df_original_trainval, df_original_test], train_fract, seed = 42)
    intersection_train_test = pd.merge(df_train, df_test, how = 'inner')
    assert(intersection_train_test.empty)

def test_intersection_split_twodfs_train_val():
    '''
    Test function for split_two_dataframes.
    In this test function it's tested that the intersection, 
    after the splitting, between the train and validation 
    dataframe is empty.

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the intersection between df_train and df_val is empty.
    '''
    df_original_trainval = pd.DataFrame({'label': ['A', 'A', 'B'],
                                    'text': ['random', 'try', 'test']})
    df_original_test = pd.DataFrame({'label': ['B', 'A', 'B'],
                              'text': ['good', 'well', 'great']})
    train_fract = 0.50
    df_train, df_val, _ = split_two_dataframes([df_original_trainval, df_original_test], train_fract, seed = 42)
    intersection_train_val = pd.merge(df_train, df_val, how = 'inner')
    assert(intersection_train_val.empty)

def test_intersection_split_twodfs_val_test():
    '''
    Test function for split_two_dataframes.
    In this test function it's tested that the intersection, 
    after the splitting, between the test and validation 
    dataframe is empty.

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the intersection between df_test and df_val is empty.
    '''
    df_original_trainval = pd.DataFrame({'label': ['A', 'A', 'B'],
                                    'text': ['random', 'try', 'test']})
    df_original_test = pd.DataFrame({'label': ['B', 'A', 'B'],
                              'text': ['good', 'well', 'great']})
    train_fract = 0.80
    _, df_val, df_test = split_two_dataframes([df_original_trainval, df_original_test], train_fract, seed = 42)
    intersection_val_test = pd.merge(df_val, df_test, how = 'inner')
    assert(intersection_val_test.empty)

def test_union_trainval_equal_firstdataframe():
    '''
    Test function for split_two_dataframes.
    In this test function it's tested that the union, 
    after the splitting, of the train and validation 
    dataframe is equal to the df_train_valid dataframe.

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the union of the train and validation dataframe is 
            equal to df_train_valid.
    '''
    df_original_trainval = pd.DataFrame({'label': ['A', 'A', 'B'],
                                    'text': ['random', 'try', 'test']})
    df_original_test = pd.DataFrame({'label': ['B', 'A', 'B'],
                              'text': ['good', 'well', 'great']})
    train_fract = 0.80
    df_train, df_valid, _ = split_two_dataframes([df_original_trainval, df_original_test], train_fract, seed = 42)
    df_trainval = pd.merge(df_train, df_valid, how = 'outer')
    # resort the value: both for union and the original dataframe
    df_trainval = df_trainval.sort_values(by=df_trainval.columns.tolist()).reset_index(drop=True)
    df_original_trainval = df_original_trainval.sort_values(by=df_original_trainval.columns.tolist()).reset_index(drop=True)
    assert(df_trainval.equals(df_original_trainval))


@given(train_fract = st.floats(min_value = 0.01, max_value = 0.99))
def test_dimension_traindf_from_twodfs_with_different_fract(train_fract):
    '''
    Test function for split_two_dataframes.
    In this test function it's tested that the train dataframe (df_train)
    has the correct dimension given by the train fraction, which is 
    generated by a strategy with value between 0.01 and 0.99.

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the dimension of df_train (train dataframe) is correct 
            (given by train fraction).
    '''
    df_original_trainval = pd.DataFrame({'text': ['a', 'b', 'c', 'd', 'e']})
    df_original_test = pd.DataFrame({'text': ['f', 'g', 'h', 'i', 'l']})

    df_train, df_val, _ = split_two_dataframes([df_original_trainval, df_original_test], train_fract, seed = 42)                                                
    assert(len(df_train) == 
                round(train_fract * (len(df_train) + len(df_val))))


@given(train_fract = st.floats(min_value = 0.01, max_value = 0.99))
def test_dimension_valdf_from_twodfs_with_different_fract(train_fract):
    '''
    Test function for split_two_dataframes.
    In this test function it's tested that the validation dataframe (df_val)
    has the correct dimension, which is length of the original dataframe minus
    the length of df_train. The train fraction to obtain df_train is generated
    by a strategy with value between 0.01 and 0.99.

    Given:
    ======
    train_frac: float
                Train fraction number that will be used as train fraction
                for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    =======
            if the dimension of df_val (train dataframe) is correct 
            (length of df_train_valid - length of df_train).
    '''
    df_original_trainval = pd.DataFrame({'text': ['a', 'b', 'c', 'd', 'e']})
    df_original_test = pd.DataFrame({'text': ['f', 'g', 'h', 'i', 'l']})

    df_train, df_val, _ = split_two_dataframes([df_original_trainval, df_original_test], train_fract, seed = 42)                                                
    assert(len(df_val) == len(df_original_trainval) - len(df_train))


def test_negative_fraction_twodfs():
    '''
    Test function for split_two_dataframes.
    In this test function it's tested what's the result if 
    the fraction passed as parameter to the function is negative.
    We expect that a ValueError is raised.

    Given:
    ======
    negative_fract: negative number
                    Negative fraction number that will be used as fraction
                    for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    ======
            if it's raised a ValueError, since the user passed a negative
            number as fraction.
    '''
    negative_fract = -0.2
    df_original_trainval = pd.DataFrame({'text': ['a', 'b', 'c']})
    df_original_test = pd.DataFrame({'text': ['f', 'g', 'h']})
    with unittest.TestCase.assertRaises(unittest.TestCase,
                                        expected_exception = ValueError):
        dfs_error = split_two_dataframes([df_original_trainval, df_original_test], negative_fract,  seed = 42)


def test_fraction_greather_than_one_twodfs():
    '''
    Test function for split_two_dataframes.
    In this test function it's tested what's the result if 
    the fraction passed as parameter to the function is greater than one.
    We expect that a ValueError is raised.

    Given:
    ======
    fract_greater_than_one: number greater than one
                            Fraction number greater than one,
                            that will be used as fraction for the function.

    df_train_valid, df_only_test: pd.DataFrame
                                  Two dataframes that represent the data
                                  and will be split by the function in 
                                  three different dataframes (train, 
                                  validation and test).

    Tests:
    ======
            if it's raised a ValueError, since the user passed a fraction number
            greater than one.
    '''
    fraction_greather_than_one = 1.2
    df_original_trainval = pd.DataFrame({'text': ['a', 'b', 'c']})
    df_original_test = pd.DataFrame({'text': ['f', 'g', 'h']})
    with unittest.TestCase.assertRaises(unittest.TestCase,
                                        expected_exception = ValueError):
        dfs_error = split_two_dataframes([df_original_trainval, df_original_test], fraction_greather_than_one,  seed = 42)

