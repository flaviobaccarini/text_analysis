'''
VISUALIZE AND EXPLORE SCRIPT
============================
Script for pre-analysis, exploration and initial visualization
of the raw data.
'''
from pathlib import Path
import pandas as pd
from text_analysis.read_write_data import read_data
from text_analysis.split_dataframe import split_dataframe
from text_analysis.cleantext import rename_columns, drop_empty_rows
from text_analysis.preanalysis import info_data, plot_label_distribution
from text_analysis.preanalysis import word_count_text, char_count_text
from text_analysis.preanalysis import average_word_or_chars
from text_analysis.preanalysis import plotting_word_char_count, printer_word_chars
import configparser
import sys

def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    analysis_folder = config_parse.get('ANALYSIS', 'folder_name')
    text_col_name = config_parse.get('ANALYSIS', 'text_column_name')
    label_col_name = config_parse.get('ANALYSIS', 'label_column_name')
    seed = int(config_parse.get('PREPROCESS', 'seed'))
    dataset_folder = Path('../datasets') / analysis_folder
    
    dfs_raw = read_data(dataset_folder)

    # split dataframe:
    if len(dfs_raw) !=  3:
        fractions = (float(config_parse.get('PREPROCESS', 'train_fraction')), 
                    float(config_parse.get('PREPROCESS', 'test_fraction')))
        dfs_raw = split_dataframe(dfs_raw, fractions, seed)


    df_new = []
    # standardize column names and remove empty cells
    for df in dfs_raw:
        df_new.append(drop_empty_rows(rename_columns(df, text_col_name, label_col_name)))
    
    df_train, df_val, df_test = df_new

    # print some info about data and plot the label distribution
    info_data(df_train, df_val, df_test)
    plot_label_distribution(df_train['label'], df_val['label'], df_test['label'])
    words_mean_list = []
    chars_mean_list = []

    # count the average number of words/chars for each text:
    for dataframe in (df_train, df_val, df_test):
        word_count = word_count_text(dataframe['text'])
        char_count = char_count_text(dataframe['text'])
        words_mean_list.append(average_word_or_chars(dataframe['label'], word_count))
        chars_mean_list.append(average_word_or_chars(dataframe['label'], char_count))
        dataframe['word_count'] = word_count
        dataframe['char_count'] = char_count
    
    # print the average number of words/chars for each text and label:
    printer_word_chars(words_mean_list, 'words')
    printer_word_chars(chars_mean_list, 'chars')

    # plot the average words/chars counting distribution
    df_complete = pd.concat([df_train, df_val, df_test], ignore_index=True)
    plotting_word_char_count(df_complete['label'], df_complete['word_count'], 'words')
    plotting_word_char_count(df_complete['label'], df_complete['char_count'], 'chars')

    
if __name__ == '__main__':
    main()
