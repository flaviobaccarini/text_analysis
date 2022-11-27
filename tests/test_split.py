from binary_classifier.split_dataframe import split_dataframe
from binary_classifier.split_dataframe import split_single_dataframe
from binary_classifier.split_dataframe import split_two_dataframes
import pandas as pd
import random
import string
from hypothesis import strategies as st
from hypothesis import given
import unittest


@given(number_of_rows = st.integers(10, 100))
def test_split_single_df(number_of_rows):
    '''
    Test function to prove the correct working for the split_single_dataframe function.
    '''
    df_fakes = [] 
    for phase in  ('alldata',): 
        fake_data = ({'phase': [phase for _ in range(number_of_rows)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(number_of_rows)], 
                         'id': [int(index) for index in range(number_of_rows)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(number_of_rows)]}) 
        fake_data_dataframe = pd.DataFrame(fake_data)
        df_fakes.append(fake_data_dataframe) 

    train_frac, test_frac = 0.70, 0.15
    assert(len(df_fakes) == 1) # one single dataframe
    df_split = split_single_dataframe(df_fakes[0], (train_frac, test_frac), seed = 42)

    assert(len(df_split) == 3) # three dataframes
    df_train, df_valid, df_test = df_split

    assert(len(df_train) == round(train_frac*number_of_rows))
    assert(len(df_test) == round(test_frac*number_of_rows))
    assert(len(df_valid) == number_of_rows - len(df_train) - len(df_test))

@given(number_of_rows = st.integers(500, 700),
       valid_fract = st.floats(min_value=0.01, max_value=0.3),
       test_fract = st.floats(min_value=0.01, max_value=0.3))
def test_split_two_df(number_of_rows, valid_fract, test_fract):
    '''
    Test function to prove the correct working for the split_two_dataframes function.
    '''
    train_frac = 1. - valid_fract - test_fract
    number_of_rows_trainval = round((train_frac+valid_fract)*number_of_rows)
    number_of_rows_test = number_of_rows - number_of_rows_trainval
    df_fakes = [] 
    for phase, nr_rows in  zip(('train_val', 'test'), (number_of_rows_trainval, number_of_rows_test)): 
        fake_data = ({'phase': [phase for _ in range(nr_rows)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(nr_rows)], 
                         'id': [int(index) for index in range(nr_rows)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(nr_rows)]}) 
        fake_data_dataframe = pd.DataFrame(fake_data)
        df_fakes.append(fake_data_dataframe) 

    assert(len(df_fakes) == 2) # two dataframes
    df_split = split_two_dataframes(df_fakes, train_frac, seed = 42)

    assert(len(df_split) == 3) # three dataframes
    df_train, df_valid, df_test = df_split

    assert(len(df_train) == round(train_frac*number_of_rows_trainval))
    assert(len(df_valid) == number_of_rows_trainval - len(df_train))
    assert(df_test.equals(df_fakes[1]))



def test_split_df_single_dataset():
    '''
    Test function to prove the correct working for the split_dataframe function for
    only one dataset provided.
    '''
    df_fakes = []
    nr_of_tot_rows = 100
    for phase in ('single_dataset',): 
        fake_data = ({'phase': [phase for _ in range(nr_of_tot_rows)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(nr_of_tot_rows)], 
                         'id': [int(index) for index in range(nr_of_tot_rows)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(nr_of_tot_rows)]}) 
        fake_data_dataframe = pd.DataFrame(fake_data)
        df_fakes.append(fake_data_dataframe) 


    assert(len(df_fakes) == 1) # one dataframe 

    train_frac, test_frac = 0.70, 0.15
    df_split = split_dataframe(df_fakes, (train_frac, test_frac), seed = 42)
    
    #df_split[0] == df_train, df_split[1] == df_valid, df_split[2] == df_test
    assert(len(df_split) == 3) # split in three dataframes
    assert(len(df_split[0]) == round(train_frac * (len(df_split[0]) + len(df_split[1]) + len(df_split[2]) )) )
    assert(len(df_split[2]) == round(test_frac * (len(df_split[0]) + len(df_split[1]) + len(df_split[2]) )) )
    assert(len(df_split[1]) == nr_of_tot_rows - len(df_split[0]) - len(df_split[2]))


def test_split_df_two_df_correct():
    '''
    Test function to prove the correct working for the split_dataframe function for
    two datasets provided in the correct condition. 
    '''
    df_fakes = []
    nr_of_tot_train_val = 1000
    nr_of_rows_test = 100
    for phase, nr_rows in zip(('train_val', 'test'), (nr_of_tot_train_val, nr_of_rows_test)): 
        fake_data = ({'phase': [phase for _ in range(nr_rows)], 
                         'tweet': [ ''.join(random.choices(string.ascii_uppercase + string.digits, k = 10)) for _ in range(nr_rows)], 
                         'id': [int(index) for index in range(nr_rows)], 
                         'label': [random.choice(['real', 'fake']) for _ in range(nr_rows)]}) 
        fake_data_dataframe = pd.DataFrame(fake_data)
        df_fakes.append(fake_data_dataframe) 

    assert(len(df_fakes) == 2) # two dataframes

    train_frac = 0.8
    fractions = (train_frac,)

    df_split = split_dataframe(df_fakes, fractions, seed = 42)
    
    #df_split[0] == df_train, df_split[1] == df_valid, df_split[2] == df_test
    assert(len(df_split) == 3) # split in three dataframes
    assert(len(df_split[0]) == round(train_frac * (len(df_split[0]) + len(df_split[1]))) )
    assert(len(df_split[1]) == nr_of_tot_train_val - len(df_split[0]))
    assert(df_split[2].equals(df_fakes[1]) )

def test_wrong_frac():
    fractions = (-0.2, 0.2)
    df_fakes = [pd.DataFrame(), pd.DataFrame()]
    seed = 42

    with unittest.TestCase.assertRaises(unittest.TestCase,
                                        expected_exception = ValueError):
        dfs_error = split_dataframe(df_fakes, fractions, seed)

    fractions = (0.2, 1.2)

    with unittest.TestCase.assertRaises(unittest.TestCase,
                                        expected_exception = ValueError):
        dfs_error = split_dataframe(df_fakes, fractions, seed)

