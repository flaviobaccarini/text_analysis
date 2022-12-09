'''
VISUALIZE AND EXPLORE MODULE
=============================
In this module some functions are defined for the visualization,
pre-analysis and initial exploration of the data.
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

def info_data(train_ds: pd.DataFrame,
              val_ds: pd.DataFrame,
              test_ds: pd.DataFrame) -> None:
    '''
    This function prints the basic informations 
    for each different dataframe (train, validation, test).
    At the beginning, the head of the train dataframe is printed.
    Then the informations about all the dataframes are printed.
    Finally, three different text strings randomly chosen from the 
    train datased are printed.

    Parameters:
    ============
    train_ds: pandas.DataFrame
              The train dataframe containing all the initial informations.

    val_ds: pandas.DataFrame
              The validation dataframe containing all the initial informations.
    
    test_ds: pandas.DataFrame
              The test dataframe containing all the initial informations.
    '''

    print("First five rows of train dataset\n" + "="*40)
    print(train_ds.head())

    print("\nDescription of train dataset\n" + "="*40)
    description_train_ds = train_ds.info()

    print("\nDescription of validation dataset\n" + "="*40)
    description_val_ds = val_ds.info()

    print("\nDescription of test dataset\n" + "="*40)
    description_test_ds = test_ds.info()

    three_text_data = train_ds['text'].sample(n = 3)
    print("\nHere some text data:\n" + "="*40)
    for text in three_text_data:
        print(text)
        print("\n" + "="*40)

    

def plot_label_distribution(label_train: pd.Series,
                            label_val: pd.Series,
                            label_test: pd.Series,) -> None:
    '''
    This function plots the label distribution for each different dataframe.
    In addition to the plot, the function prints also the number of label distribution
    for each different dataframe.

    Parameters:
    ============
    label_train: pandas.Series
                 The labels from the train dataset.

    label_val: pandas.Series
               The labels from the validation dataset.
    
    label_test: pandas.Series
                The labels from the test dataset.
    '''
    sns.set(font_scale=1.4)
    fig, ax = plt.subplots()

    value_counts_label = label_train.value_counts()
    sns.barplot(x = value_counts_label.index, y = value_counts_label, ax = ax)
    print("\nTrain dataset label distribution:\n{0}".format(value_counts_label))
    ax.set_xlabel("Label", labelpad=12)
    ax.set_ylabel("Count of labels", labelpad=12)
    ax.set_title("Train dataset label distribution", y=1.02)

    fig1, ax1 = plt.subplots()
    value_counts_label = label_val.value_counts()
    sns.barplot(x = value_counts_label.index, y = value_counts_label, ax = ax1)
    print("\nValid dataset label distribution:\n{0}".format(value_counts_label))
    ax1.set_xlabel("Label", labelpad=12)
    ax1.set_ylabel("Count of labels", labelpad=12)
    ax1.set_title("Valid dataset label distribution", y=1.02)

    fig2, ax2 = plt.subplots()
    value_counts_label = label_test.value_counts()
    sns.barplot(x = value_counts_label.index, y = value_counts_label, ax = ax2)
    print("\nTest dataset label distribution:\n{0}".format(value_counts_label))
    ax2.set_xlabel("Label", labelpad=12)
    ax2.set_ylabel("Count of labels", labelpad=12)
    ax2.set_title("Test dataset label distribution", y=1.02)

    plt.show()


def word_count_text(text: ArrayLike) -> list[int]:
    '''
    This function counts how many words there are for each text string
    to compare the number of words difference between different labels.

    Parameters:
    ============
    text:     1-D array-like[str]
              The sequence of the text to count the number of words. 

    Raises:
    ========
    ValueError: if the text sequence is empty.

    Return:
    ========
    word_count: list[int]
                The list contains the number of words for each single text.
    '''
    if len(text) == 0:
        raise ValueError("The text sequence passed is empty")
    list_text = list(text)
    word_count = [len(str(words).split()) for words in list_text]
    return word_count


def printer_word_chars(all_dicts_avg_labels: list[dict],
                              unit_of_measure: str) -> None:
    '''
    This function prints the average number of words/chars
    for each different dataframe and label.

    Parameters:
    ============
    all_dicts_avg_labels: list[dict]
                          This list contain three different dictionaries
                          (training, validation, test).
                          Dictionary keys represent unique labels 
                          and each key is mapped to the number of
                          words/chars average for that label.
    
    unit_of_measure: str 
                     This string could be "chars" or "words" and it represents
                     the "unit of measure" for the average number inside the dictionary.
    '''
    train_word_char_mean_dict, val_word_char_mean_dict, test_word_char_mean_dict = all_dicts_avg_labels
    for key in train_word_char_mean_dict:
        print(f'{key} labels length (average {unit_of_measure}):'
              f'training {train_word_char_mean_dict[key]:.1f}', 
              f'validation {val_word_char_mean_dict[key]:.1f}', 
              f'test {test_word_char_mean_dict[key]:.1f}')

def average_word_or_chars(labels: ArrayLike, 
                          word_or_char_count: ArrayLike) -> dict:
    '''
    This function computes the average number of words/chars
    for each different label.

    Parameters:
    ============
    labels: 1-D array-like
            Sequence containing the labels for each single text.
    
    word_or_char_count: 1-D array-like[int]
                        Sequence containing the number of words/chars
                        for each single text.
                     
    Raises:
    ========
    ValueError
        if labels sequence is empty 
    ValueError
        if word_or_char_count sequence is empty 

    Returns:
    =========
    unique_labels_dict: dict
                        Each single key of this dictionary is a unique label.
                        All the labels are mapped to the average number of words/chars
                        computed on the text referred to that label.
    '''
    if len(labels) == 0:  
        raise ValueError('The labels sequence passed is empty')
    if len(word_or_char_count) == 0: 
        raise ValueError('The word_or_char_count sequence passed is empty')

    labels_series = pd.Series(labels)
    unique_labels = labels_series.unique()
    unique_labels_dict = dict.fromkeys(unique_labels, None)
    df_word_char_count = pd.DataFrame({'label': labels_series,
                                        'count': word_or_char_count})
    
    for key_lab in unique_labels_dict:
        unique_labels_dict[key_lab] = df_word_char_count[df_word_char_count['label'] == key_lab]['count'].mean()
    
    return unique_labels_dict


def char_count_text(text: ArrayLike) -> list[int]:
    '''
    This function counts how many characters there are for each text in order 
    to compare the difference in the number of characters between different labels.

    Parameters:
    ============
    text:   1-D array-like[str]
            The sequence of the text to count the number of characters. 

    Raises:
    =======
    ValueError: if the text sequence is empty.

    Returns:
    ========
    char_count: list[int]
                The list contains the number of characters for each single text.
    '''
    if len(text) == 0:
        raise ValueError("The text sequence passed is empty")
    list_text = list(text)
    char_count = [len(str(chars)) for chars in list_text]
    return char_count


def plotting_word_char_count(labels: ArrayLike,
                             word_char_count: ArrayLike,
                             unit_of_measure: str) -> None:
    '''
    This function plots the average number of words/characaters for each dataframe
    highlighting the difference in numbers between different labels.

    Parameters:
    ============
    labels: 1-D array-like
            Sequence that contains all the labels for the data.
    
    word_char_count: 1-D array-like[int]
                     Sequence that contains all the word/character
                     counts for each single text data.

    unit_of_measure: str
                     String that could be "chars"/"characters" or
                     "words" and represents if the numbers inside 
                     word_char_count are referred to characters
                     or words. 

    '''  
    word_char_count_df = pd.DataFrame({'label': labels, 'count': word_char_count})
    if unit_of_measure == 'chars':
        color = 'green'
    else:
        color = 'red'
    # PLOTTING
    sns.set(font_scale=1.4)
    
    fig, ax=plt.subplots(1,len(labels.unique()),figsize=(10,4))
    for index, key in enumerate(labels.unique()):
        counts = word_char_count_df[word_char_count_df['label'] == key]['count']
        ax[index].hist(counts, color = color, range = (0, np.median(counts)*1.5))
        ax[index].set_title(f'{key} labels {unit_of_measure} distribution', y = 1.02)
        ax[index].set_xlabel(f'number of {unit_of_measure}')
        ax[index].set_ylabel(f'count #{unit_of_measure}')
    plt.show()

