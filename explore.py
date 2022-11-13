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
import seaborn as sns
import matplotlib.pyplot as plt
from read_write_data import read_data, clean_dataframe, split_dataframe
import configparser
import sys

#TODO CERCA DI RENDERE LE FUNZIONI TUTTE UN PO' PIÙ GENERICHE... ---> QUESTO MODULO OK (MAGARI NON PERFETTO MA MEGLIO); CONTROLLA LA FUNZIONE DI READING

#TODO CONTROLLA LA DOCUMENTAZIONE DI TUTTE LE FUNZIONI ORA
def info_data(train_ds: pd.DataFrame,
              val_ds: pd.DataFrame,
              test_ds: pd.DataFrame) -> None:
    '''
    This function prints the basic informations for each different dataframe (train, validation, test).
    At the beginning, the head of the train dataframe is printed.
    Then the informations about all the dataframes are printed.
    Finally, three different tweets randomly chosen from the train datased are printed.

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
    In addition to the plot, it will print also the number of label distribution
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


def word_count_twitter(tweets: pd.Series(str)) -> list[int]:
    '''
    This function counts how many words there are for each tweet in order to compare the difference
    in the number of words between real news and fake news regarding COVID.

    Parameters:
    ============
    tweets: pandas.Series(str) or list[str] or tuple[str]
              The sequence of the tweets to count the number of words. 

    Return:
    ========
    word_count: list[int]
                The list containing the number of words for each single tweet.
    '''
    list_tweets = list(tweets)
    word_count = [len(str(words).split()) for words in list_tweets]
   
    return word_count


def printer_word_chars(all_dicts_avg_labels,
                              unit_of_measure) -> None:
    '''
    This function prints the average number of words for each dataframe
    comparing the difference between fake news and real news.

    Parameters:
    ============
    
    '''
    train_word_char_mean_dict, val_word_char_mean_dict, test_word_char_mean_dict = all_dicts_avg_labels
    for key in train_word_char_mean_dict:
        print(f'{key} labels length (average {unit_of_measure}):'
              f'training {train_word_char_mean_dict[key]:.1f}, validation {val_word_char_mean_dict[key]:.1f}, '
              f'test {test_word_char_mean_dict[key]:.1f}')

#TODO NEED SOME TESTING FOR THE BELOW FUNCTION
def average_word_or_chars(labels, word_or_char_count):
    unique_labels = labels.unique()
    unique_labels_dict = dict.fromkeys(unique_labels, None)
    df_word_char_count = pd.DataFrame({'label': labels, 'count': word_or_char_count})

    for key_lab in unique_labels_dict:
        unique_labels_dict[key_lab] = df_word_char_count[df_word_char_count['label'] == key_lab]['count'].mean()
    
    return unique_labels_dict


def char_count_twitter(tweets: pd.Series(str)) -> list[int]:
    '''
    This function counts how many characters there are for each tweet in order 
    to compare the difference in the number of characters between real news and fake news.

    Parameters:
    ============
    tweets: pandas.Series(str) or list[str] or tuple[str]
              The sequence of the tweets to count the number of characters. 

    Return:
    ========
    char_count: list[int]
                The list containing the number of characters for each single tweet.
    '''
    list_tweets = list(tweets)
    char_count = [len(str(chars)) for chars in list_tweets]
    return char_count


def plotting_word_char_count(labels, word_char_count, unit_of_measure) -> None:
    '''
    This function plots the average number of characters for each dataframe
    comparing the difference between fake news and real news.

    Parameters:
    ============

    '''  
    word_char_count_df = pd.DataFrame({'label': labels, 'count': word_char_count})
    if unit_of_measure == 'chars':
        range = (0, 400)
        color = 'green'
    else:
        range = (0, 50)
        color = 'red'
    # PLOTTING
    sns.set(font_scale=1.4)
    
    fig, ax=plt.subplots(1,len(labels.unique()),figsize=(10,4))
    for index, key in enumerate(labels.unique()):
        counts = word_char_count_df[word_char_count_df['label'] == key]['count']
        ax[index].hist(counts, color = color, range = range)
        ax[index].set_title(f'{key} labels {unit_of_measure} distribution', y = 1.02)
        ax[index].set_xlabel(f'number of {unit_of_measure}')
        ax[index].set_ylabel(f'count #{unit_of_measure}')
    plt.show()


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
    dataset_folder = 'datasets'
    
    dfs_raw = read_data(dataset_folder, analysis_folder)
    if len(dfs_raw) !=  3:
        fractions = (float(config_parse.get('PREPROCESS', 'train_fraction')), 
                    float(config_parse.get('PREPROCESS', 'val_fraction')),
                    float(config_parse.get('PREPROCESS', 'test_fraction')))
        dfs_raw = split_dataframe(dfs_raw, fractions)

    column_names = (config_parse.get('PREPROCESS', 'column_name_text'), 
                    config_parse.get('PREPROCESS', 'column_name_label'))

    df_train, df_val, df_test = clean_dataframe(dfs_raw, column_names)
    
    explore(df_train, df_val, df_test)

if __name__ == '__main__':
    main()
