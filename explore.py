'''
VISUALIZE AND EXPLORE MODULE
=============================
In this module some functions are defined for the visualization, analysis and exploration of the data.

In particular, the info_data function prints the information about the three datasets (train, validation and test dataframes). 
The plot_label_distribution function plots the label distribution in order to check if it's balanced or not.
The word_count_twitter function counts how many words per each tweet are contained, while
the word_count_printer function prints how many words per each tweet, then
the plotting_word_count function plots how many words per each tweet depending on the label.
The char_count_twitter function counts how many chars per each tweet are contained, while
the char_count_printer function prints how many chars per each tweet, then
the plotting_char_count function plots how many chars per each tweet depending on the label.
Finally the explore function executes all the previous discussed functions. 
'''
from pathlib import Path
import pandas as pd
from binary_classifier.read_write_data import read_data, split_dataframe
from binary_classifier.cleantext import rename_columns, drop_empty_rows
from binary_classifier.preanalysis import info_data, plot_label_distribution
from binary_classifier.preanalysis import word_count_twitter, char_count_twitter
from binary_classifier.preanalysis import average_word_or_chars
from binary_classifier.preanalysis import plotting_word_char_count, printer_word_chars
import configparser
import sys


def explore(df_train, df_val, df_test):
    '''
    This function executes all the previous functions in order to 
    visualize, describe, analyze and explore the data.
    Parameters:
    ============

    '''
    info_data(df_train, df_val, df_test)
    plot_label_distribution(df_train['label'], df_val['label'], df_test['label'])
    words_mean_list = []
    chars_mean_list = []
    for dataframe in (df_train, df_val, df_test):
        word_count = word_count_twitter(dataframe['text'])
        char_count = char_count_twitter(dataframe['text'])
        words_mean_list.append(average_word_or_chars(dataframe['label'], word_count))
        chars_mean_list.append(average_word_or_chars(dataframe['label'], char_count))
        dataframe['word_count'] = word_count
        dataframe['char_count'] = char_count
    printer_word_chars(words_mean_list, 'words')
    printer_word_chars(chars_mean_list, 'chars')

    df_complete = pd.concat([df_train, df_val, df_test], ignore_index=True)
    plotting_word_char_count(df_complete['label'], df_complete['word_count'], 'words')
    plotting_word_char_count(df_complete['label'], df_complete['char_count'], 'chars')


def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('INPUT_OUTPUT', 'analysis')
    seed = int(config_parse.get('PREPROCESS', 'seed'))
    dataset_folder = Path('datasets') / analysis_folder
    
    dfs_raw = read_data(dataset_folder)

    if len(dfs_raw) !=  3:
        fractions = (float(config_parse.get('PREPROCESS', 'train_fraction')), 
                    float(config_parse.get('PREPROCESS', 'test_fraction')))
        dfs_raw = split_dataframe(dfs_raw, fractions, seed)

    column_names = (config_parse.get('PREPROCESS', 'column_name_text'), 
                    config_parse.get('PREPROCESS', 'column_name_label'))

    df_new = []
    for df in dfs_raw:
        df_new.append(drop_empty_rows(rename_columns(df, column_names)))
    
    df_train, df_val, df_test = df_new
    explore(df_train, df_val, df_test)
    
if __name__ == '__main__':
    main()
