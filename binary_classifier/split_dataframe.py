'''
SPLITTING MODULE
===========================
This module is dedicated for the splitting functions, useful
if the data are not already split in three different datasets.
'''
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

def split_dataframe(dataframes_list: tuple[pd.DataFrame],
                    fractions: ArrayLike,
                    seed: int) -> tuple[pd.DataFrame]:
    '''
    Function used to split the dataframe/dataframes.
    It takes as input the dataframes_list which could be
    a list of length one or two (one single dataframe, or
    two already split dataframes in train and test data);
    the fractions to split the dataframes; the seed in order 
    to have a determinitic behaviour.
    The dataframes are returned in this order: train, validation
    and test dataset.

    Parameters:
    ============
    dataframes_list: tuple[pd.DataFrame]
                     Sequence that contains one or two dataframes
                     to split. The length must be at maximum of 2.
    
    fractions: 1-D array-like[float]
               Fractions that determines how the data have
               to be split in the different dataset.
               The sequence needs to have two values.
               They have to be float number between 0 and 1.
    
    seed: number
          Number to set the seed in order to have not a completely 
          randomic behaviour.
    
    Raises:
    ========
    ValueErorr: if one fractions is composed by numbers not in the interval [0, 1]

    Returns:
    =========
    df_train: pd.DataFrame
              Train dataframe after splitting operation.

    df_valid: pd.DataFrame
              Validation dataframe after splitting operation.
              
    df_test: pd.DataFrame
              Test dataframe after splitting operation.
    '''

    if (np.array(fractions) >= 1).any() or (np.array(fractions) <= 0).any():
        raise ValueError("Fractions have to be float number"
                          "between 0 and 1")

    if len(dataframes_list) == 2:
        train_frac, *_ = fractions
        df_train, df_valid, df_test = split_two_dataframes(dataframes_list,
                                                           train_frac,
                                                           seed)
    else:
        train_frac, test_frac = fractions
        df_train, df_valid, df_test = split_single_dataframe(dataframes_list[0],
                                                            (train_frac, test_frac),
                                                             seed)

    df_train = df_train.reset_index(drop = True)
    df_valid = df_valid.reset_index(drop = True)
    df_test = df_test.reset_index(drop = True)

    return df_train, df_valid, df_test


def split_single_dataframe(single_df: pd.DataFrame,
                           fractions: ArrayLike, 
                           seed: int) -> tuple[pd.DataFrame]:
    '''
    Helper function for splitting data:
    case where only one single dataframe is provided.
    This function takes as input one single dataframe and the fractions
    to divide the single initial dataset in the three final datasets.
    The dataframes are returned in this order: train, validation
    and test dataset.

    Parameters:
    ============
    single_df: pd.DataFrame
               Dataframe that contains all the data.

    fractions: 1-D array-like[float]
               Divide the data in the three final dataset
               accordingly these fraction numbers.
               Sequence of length two (train and test fractions).
    
    seed: number
          Number to set the seed in order to have not a completely 
          randomic behaviour.

    returns:
    ==========
    df_train: pd.DataFrame
              Train dataframe after splitting operation.

    df_valid: pd.DataFrame
              Validation dataframe after splitting operation.
              
    df_test: pd.DataFrame
              Test dataframe after splitting operation.
    '''
    train_frac, test_frac = fractions
    df_test = single_df.sample(frac = test_frac, random_state = seed)
    df_train_val = single_df.drop(df_test.index)
    df_train = df_train_val.sample(n = round(train_frac*len(single_df)), random_state = seed)
    df_valid = df_train_val.drop(df_train.index)
    return df_train, df_valid, df_test

def split_two_dataframes(dataframes: tuple[pd.DataFrame],
                         train_frac: float, 
                         seed: int) -> tuple[pd.DataFrame]:
    '''
    Helper function for splitting data:
    case where two different dataframes are provided.
    This function takes as input a list containing two different
    dataframes. The first one is considered as the train dataset
    and so it will split in train and validation. The second one
    is considered as the test dataset.
    The other important input parameter is the train_frac,
    which represents the train fraction. The train dataframe will
    have a number of samples equal to the train fraction times 
    the total number of samples inside the original dataframe (first
    dataframe in the dataframes list given as input).
    The dataframes are returned in this order: train, validation
    and test dataset.

    Parameters:
    ============
    dataframes: tuple(pd.DataFrame)
                Tuple that contains two dataframes.
                The first dataframe corresponds to the train dataset,
                while the second dataframe corresponds to the test dataset.

    train_frac: float
                Divide the first dataframe in two dataframes;
                if train_frac is 0.90, the first new dataframe (train)
                will have the 90% of the data from the initial
                dataframe, while the validation dataframe will have
                the remaining 10% of the data.
                
    seed: number
          Number to set the seed in order to have not a completely 
          randomic behaviour.

    returns:
    ==========
    df_train: pd.DataFrame
              Train dataframe after splitting operation.

    df_valid: pd.DataFrame
              Validation dataframe after splitting operation.
              
    df_test: pd.DataFrame
              Test dataframe after splitting operation.
    '''
    n_frac = round( len(dataframes[0]) * train_frac)
    df_train = dataframes[0].sample(n = n_frac, random_state = seed)
    df_valid = dataframes[0].drop(df_train.index)
    df_test = dataframes[1]
    return df_train, df_valid, df_test

