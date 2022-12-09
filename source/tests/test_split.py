'''
Test functions for the splitting module.
'''
from text_analysis.split_dataframe import split_dataframe
from text_analysis.split_dataframe import split_single_dataframe
from text_analysis.split_dataframe import split_two_dataframes
from tests.test_input_output import create_fake_data
import pandas as pd
from hypothesis import strategies as st
from hypothesis import given
import unittest


def test_split_df_single_dataset():
    '''
    Test function to test the behaviour of split_dataframe function.
    The function takes as input a single dataframe.
    The split dataframes are returned in this orderd: train, validation and test
    dataframe.    
    '''
    # create some fake data:
    phases = ('single_dataframe')
    nr_of_tot_rows = 500
    df_fakes = create_fake_data(phases, (nr_of_tot_rows,))

    assert(len(df_fakes) == 1) # one dataframe 

    # initialize the train and test fraction
    train_frac, test_frac = 0.70, 0.15
    # split
    df_split = split_dataframe(df_fakes, (train_frac, test_frac), seed = 42)
    
    #df_split[0] == df_train, df_split[1] == df_valid, df_split[2] == df_test
    # verify that each dataframe has the correct dimensionality.    
    assert(len(df_split) == 3) # split in three dataframes
    assert(len(df_split[0]) == 
                round(train_frac * (len(df_split[0]) + len(df_split[1]) + len(df_split[2]))))
    assert(len(df_split[2]) == 
                round(test_frac * (len(df_split[0]) + len(df_split[1]) + len(df_split[2]))))
    assert(len(df_split[1]) == 
                nr_of_tot_rows - len(df_split[0]) - len(df_split[2]))

@given(number_of_rows = st.integers(min_value = 5, max_value = 10000))
def test_split_single_df(number_of_rows):
    '''
    Test function to test the behaviour of split_single_dataframe function.
    The function takes as input a single dataframe.
    Two fractions are needed: the train and test fraction (then the validation 
    fraction is simply 1 - (train_frac + test_frac)).
    The split dataframes are returned in this orderd: train, validation and test
    dataframe.

    @given:
    =======
    number_of_rows: int
                    Number of rows for the single dataframe.
    '''
    # crate some fake data
    phases = ('alldata')
    df_fakes = create_fake_data(phases, (number_of_rows, ))
   
    # initialize the train and test fraction
    train_frac, test_frac = 0.70, 0.15
    assert(len(df_fakes) == 1) # one single dataframe
    
    # split dataframe
    df_split = split_single_dataframe(df_fakes[0], (train_frac, test_frac), seed = 42)

    assert(len(df_split) == 3) # three dataframes
    
    # verify that each dataframe has the correct dimensionality.
    df_train, df_valid, df_test = df_split
    assert(len(df_train) == round(train_frac*number_of_rows))
    assert(len(df_test) == round(test_frac*number_of_rows))
    assert(len(df_valid) == number_of_rows - len(df_train) - len(df_test))


@given(train_fract = st.floats(min_value=0.01, max_value=0.45),
       test_fract = st.floats(min_value=0.01, max_value=0.45))
def test_split_single_df_fractions(train_fract, test_fract):
    '''
    Test function to test the behaviour of split_singe_dataframe function.
    The function takes as input a sequence of one single dataframe.
    Two fraction are needed: the train and test fractions.
    The split dataframes are returned in this orderd: train, validation and test
    dataframe.

    @given:
    =========
    train_fract: float
                 Train fraction
    
    test_frac: float
               Test fraction
    '''
    number_of_rows = 5000
    # generate fake date
    phases = ('single')
    df_fakes = create_fake_data(phases, [number_of_rows])

    assert(len(df_fakes) == 1) # one single dataframe
    df_split = split_single_dataframe(df_fakes[0], (train_fract, test_fract), seed = 42)

    # check the splitting
    assert(len(df_split) == 3) # three dataframes
    df_train, df_valid, df_test = df_split

    # check the dimensionality of the dataframes
    assert(len(df_train) == round(train_fract*number_of_rows))
    assert(len(df_test) == round(test_fract*number_of_rows))
    assert(len(df_valid) == number_of_rows - len(df_train) - len(df_test))



def test_split_df_two_df_correct():
    '''
    Test function to test the behaviour of split_dataframe function for
    two datasets.
    The function takes as input a list of two dataframes.
    The first one is assumed to be the train dataset and it will be split
    in train and validation dataset.
    The second one is assumed to be the test dataset and it won't be modified.
    The split dataframes are returned in this orderd: train, validation and test
    dataframe.    
    The size of the train dataset will be equal to the train fraction times the 
    total number of samples inside the first dataframe from the list.
    The remaining samples from the first dataframe will be stored inside 
    the validation dataset.
    '''
    # create fake data
    phases = ('train_val', 'test')
    nr_of_rows = 200
    df_fakes = create_fake_data(phases, (nr_of_rows, nr_of_rows))

    assert(len(df_fakes) == 2) # two dataframes

    # we need only the train fraction
    # because the test dataset is already given
    train_frac = 0.8
    # need to create a tuple, because split_dataframe takes a tuple as input
    fractions = (train_frac,)

    df_split = split_dataframe(df_fakes, fractions, seed = 42)
    
    # df_split[0] == df_train, df_split[1] == df_valid, df_split[2] == df_test
    assert(len(df_split) == 3) # split in three dataframes
    # verify the dimensionality:
    assert(len(df_split[0]) == round(train_frac * (len(df_split[0]) + len(df_split[1]))) )
    assert(len(df_split[1]) == nr_of_rows - len(df_split[0]))
    assert(df_split[2].equals(df_fakes[1]) )


@given(number_of_rows = st.integers(min_value = 5, max_value = 10000))
def test_split_two_dfs_rows(number_of_rows):
    '''
    Test function to test the behaviour of split_two_dataframes function.
    The function takes as input a sequence of two dataframes.
    One fraction is needed: the train fraction.
    The split dataframes are returned in this orderd: train, validation and test
    dataframe.

    @given:
    =======
    number_of_rows: int
                    Number of rows for the single dataframe.
    '''
    # crate some fake data
    phases = ('train', 'test')
    df_fakes = create_fake_data(phases, (number_of_rows, number_of_rows))
   
    # initialize the train fraction
    train_frac = 0.70
    assert(len(df_fakes) == 2) # two dataframes
    
    # split dataframe
    df_split = split_two_dataframes(df_fakes, train_frac, seed = 42)

    assert(len(df_split) == 3) # three dataframes
    
    # verify that each dataframe has the correct dimensionality.
    df_train, df_valid, df_test = df_split
    assert(len(df_train) == round(train_frac*number_of_rows))
    assert(df_test.equals(df_fakes[1]))
    assert(len(df_valid) == number_of_rows - len(df_train))


@given(train_fract = st.floats(min_value=0.01, max_value=0.45),
       test_fract = st.floats(min_value=0.01, max_value=0.45))
def test_split_two_df(train_fract, test_fract):
    '''
    Test function to test the behaviour of split_two_dataframes function.
    The function takes as input a sequence of two dataframes.
    One fraction is needed: the train fraction.
    The second dataframe from the list will not be split (it is assumed to be
    the test dataset). The first dataframe of the list will be split in two dataframes:
    the first one is the train dataframe with train fraction of samples from the 
    original dataframe, while the second one is the validation dataframe with the 
    remaining samples from the original dataframe.
    The split dataframes are returned in this orderd: train, validation and test
    dataframe.

    @given:
    =========
    train_fract: float
                 Train fraction
    
    test_frac: float
               Test fraction
    '''
    number_of_rows = 5000
    valid_fract = 1. - train_fract - test_fract
    # divide number of rows between train/valid dataset and test dataset
    number_of_rows_trainval = round((train_fract+valid_fract)*number_of_rows)
    number_of_rows_test = number_of_rows - number_of_rows_trainval
    # generate fake date
    phases = ('trainval', 'test')
    rows = (number_of_rows_trainval, number_of_rows_test)
    df_fakes = create_fake_data(phases, rows)

    assert(len(df_fakes) == 2) # two dataframes
    df_split = split_two_dataframes(df_fakes, train_fract, seed = 42)

    # check the splitting
    assert(len(df_split) == 3) # three dataframes
    df_train, df_valid, df_test = df_split

    # check the dimensionality of the dataframes
    assert(len(df_train) == round(train_fract*number_of_rows_trainval))
    assert(len(df_valid) == number_of_rows_trainval - len(df_train))
    assert(df_test.equals(df_fakes[1]))

def test_wrong_frac():
    '''
    Test function to test the behaviour of split_dataframe when a negative
    fraction or a fraction greather than one is passed as input.
    The function raises a ValueError. 
    Fractions must be two numbers within the interval (0, 1).
    '''
    # try with negative value:
    fractions = (-0.2, 0.2)
    df_fakes = [pd.DataFrame(), pd.DataFrame()]
    seed = 42

    # we expect an ValueError
    with unittest.TestCase.assertRaises(unittest.TestCase,
                                        expected_exception = ValueError):
        dfs_error = split_dataframe(df_fakes, fractions, seed)

    # try with fraction greater than 1
    fractions = (0.2, 1.2)

    # we expect a ValueError
    with unittest.TestCase.assertRaises(unittest.TestCase,
                                        expected_exception = ValueError):
        dfs_error = split_dataframe(df_fakes, fractions, seed)

