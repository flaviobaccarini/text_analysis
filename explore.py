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
from read_write_data import read_data
import configparser
import sys

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

    print("\nSome tweets extracted from train dataset\n" + "="*40)
    sampled_ds = train_ds.sample(n = 3, ignore_index = True, random_state = 42)
    print(sampled_ds['tweet'][0] + '\n\n' + 
          sampled_ds['tweet'][1] + '\n\n' +
          sampled_ds['tweet'][2])
    

def plot_label_distribution(train_ds: pd.DataFrame,
                            val_ds: pd.DataFrame,
                            test_ds: pd.DataFrame) -> None:
    '''
    This function plots the label distribution for each different dataframe.
    In addition to the plot, it will print also the number of label distribution
    for each different dataframe.

    Parameters:
    ============
    train_ds: pandas.DataFrame
              The train dataframe containing all the initial informations.

    val_ds: pandas.DataFrame
              The validation dataframe containing all the initial informations.
    
    test_ds: pandas.DataFrame
              The test dataframe containing all the initial informations.
    '''
    sns.set(font_scale=1.4)
    fig, ax = plt.subplots()

    value_counts_label = train_ds['label'].value_counts()
    sns.barplot(x = value_counts_label.index, y = value_counts_label, ax = ax)
    print("\nTrain dataset label distribution:\n{0}".format(value_counts_label))
    ax.set_xlabel("COVID19 News", labelpad=14)
    ax.set_ylabel("Count of Real/Fake News", labelpad=14)
    ax.set_title("Count of Real/Fake News for training set", y=1.02)

    fig1, ax1 = plt.subplots()
    value_counts_label = val_ds['label'].value_counts()
    sns.barplot(x = value_counts_label.index, y = value_counts_label, ax = ax1)
    print("\nValid dataset label distribution:\n{0}".format(value_counts_label))
    ax1.set_xlabel("COVID19 News", labelpad=14)
    ax1.set_ylabel("Count of Real/Fake News", labelpad=14)
    ax1.set_title("Count of Real/Fake News for valid set", y=1.02)

    fig2, ax2 = plt.subplots()
    value_counts_label = test_ds['label'].value_counts()
    sns.barplot(x = value_counts_label.index, y = value_counts_label, ax = ax2)
    print("\nTest dataset label distribution:\n{0}".format(value_counts_label))
    ax2.set_xlabel("COVID19 News", labelpad=14)
    ax2.set_ylabel("Count of Real/Fake News", labelpad=14)
    ax2.set_title("Count of Real/Fake News for test set", y=1.02)

    plt.show()


def word_count_twitter(train_ds: pd.DataFrame,
                        val_ds: pd.DataFrame,
                        test_ds: pd.DataFrame) -> tuple[pd.DataFrame]:
    '''
    This function counts how many words there are for each tweet in order to compare the difference
    in the number of words between real news and fake news regarding COVID.

    Parameters:
    ============
    train_ds: pandas.DataFrame
              The train dataframe containing all the initial informations.

    val_ds: pandas.DataFrame
              The validation dataframe containing all the initial informations.
    
    test_ds: pandas.DataFrame
              The test dataframe containing all the initial informations.

    Return:
    ========
    train_ds_with_word_count: pandas.DataFrame
                              The train dataframe containing all the initial informations and
                              the number of words for each single tweet.

    val_ds_with_word_count: pandas.DataFrame
                              The validation dataframe containing all the initial informations and
                              the number of words for each single tweet.

    test_ds_with_word_count: pandas.DataFrame
                              The test dataframe containing all the initial informations and
                              the number of words for each single tweet.
    '''
    train_ds_with_word_count = train_ds.copy()
    val_ds_with_word_count = val_ds.copy()
    test_ds_with_word_count = test_ds.copy()

    train_ds_with_word_count['word_count'] = [len(str(words).split()) for words in train_ds_with_word_count['tweet']]
    val_ds_with_word_count['word_count'] = [len(str(words).split()) for words in val_ds_with_word_count['tweet']]
    test_ds_with_word_count['word_count'] = [len(str(words).split()) for words in test_ds_with_word_count['tweet']]

    return train_ds_with_word_count, val_ds_with_word_count, test_ds_with_word_count

def word_count_printer(train_ds_with_word_count: pd.DataFrame,
                       val_ds_with_word_count: pd.DataFrame,
                       test_ds_with_word_count: pd.DataFrame) -> None:
    '''
    This function prints the average number of words for each dataframe
    comparing the difference between fake news and real news.

    Parameters:
    ============
    train_ds_with_word_count: pandas.DataFrame
              The train dataframe containing all the initial informations and
              the number of words for each single tweet.

    val_ds_with_word_count: pandas.DataFrame
              The validation dataframe containing all the initial informations and
              the number of words for each single tweet.
    
    test_ds_with_word_count: pandas.DataFrame
              The test dataframe containing all the initial informations amd
              the number of words for each single tweet.
    '''                   
    print("Real news length (average words):" 
          "training {0:.1f}, validation {1:.1f}, test {2:.1f}".format(
        train_ds_with_word_count[train_ds_with_word_count['label']=='real']['word_count'].mean(),
        val_ds_with_word_count[val_ds_with_word_count['label']=='real']['word_count'].mean(),
        test_ds_with_word_count[test_ds_with_word_count['label']=='real']['word_count'].mean()))

    print("Fake news length (average words):" 
          "training {0:.1f}, validation {1:.1f}, test {2:.1f}".format(
        train_ds_with_word_count[train_ds_with_word_count['label']=='fake']['word_count'].mean(),
        val_ds_with_word_count[val_ds_with_word_count['label']=='fake']['word_count'].mean(),
        test_ds_with_word_count[test_ds_with_word_count['label']=='fake']['word_count'].mean()))


def plotting_word_count(train_ds_with_word_count: pd.DataFrame,
                       val_ds_with_word_count: pd.DataFrame,
                       test_ds_with_word_count: pd.DataFrame) -> None:

    '''
    This function plots the average number of words for each dataframe
    comparing the difference between fake news and real news.

    Parameters:
    ============
    train_ds_with_word_count: pandas.DataFrame
              The train dataframe containing all the initial informations and
              the number of words for each single tweet.

    val_ds_with_word_count: pandas.DataFrame
              The validation dataframe containing all the initial informations and
              the number of words for each single tweet.
    
    test_ds_with_word_count: pandas.DataFrame
              The test dataframe containing all the initial informations amd
              the number of words for each single tweet.
    '''   
    # PLOTTING WORD-COUNT
    sns.set(font_scale=1.4)
    complete_ds = pd.concat([train_ds_with_word_count,
                             val_ds_with_word_count,
                             test_ds_with_word_count], ignore_index = True)

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    
    words=complete_ds[complete_ds['label']=='real']['word_count']
    ax1.hist(words,color='red', range = (0, 50))
    ax1.set_title('Real news COVID19')
    
    words=complete_ds[complete_ds['label']=='fake']['word_count']
    ax2.hist(words,color='green', range = (0, 50))
    ax2.set_title('Fake news COVID19')
    
    fig.suptitle('Words per tweet')
    plt.show()

def char_count_twitter(train_ds: pd.DataFrame,
                        val_ds: pd.DataFrame,
                        test_ds: pd.DataFrame) -> tuple[pd.DataFrame]:
    '''
    This function counts how many characters there are for each tweet in order 
    to compare the difference in the number of characters between real news and fake news.

    Parameters:
    ============
    train_ds: pandas.DataFrame
              The train dataframe containing all the initial informations.

    val_ds: pandas.DataFrame
              The validation dataframe containing all the initial informations.
    
    test_ds: pandas.DataFrame
              The test dataframe containing all the initial informations.

    Return:
    ========
    train_ds_char_count: pandas.DataFrame
                              The train dataframe containing all the initial informations and
                              the number of characters for each single tweet.

    val_ds_char_count: pandas.DataFrame
                              The validation dataframe containing all the initial informations and
                              the number of characters for each single tweet.

    test_ds_char_count: pandas.DataFrame
                              The test dataframe containing all the initial informations and
                              the number of characters for each single tweet.
    '''
    train_ds_char_count = train_ds.copy()
    val_ds_char_count = val_ds.copy()
    test_ds_char_count = test_ds.copy()

    train_ds_char_count['char_count'] = [len(str(x)) for x in train_ds_char_count['tweet'] ]
    val_ds_char_count['char_count'] = [len(str(x)) for x in val_ds_char_count['tweet'] ]
    test_ds_char_count['char_count'] = [len(str(x)) for x in test_ds_char_count['tweet'] ]

    return train_ds_char_count, val_ds_char_count, test_ds_char_count

def char_count_printer(train_ds_char_count: pd.DataFrame,
                        val_ds_char_count: pd.DataFrame,
                        test_ds_char_count: pd.DataFrame) -> None:
    '''
    This function prints the average number of characters for each dataframe
    comparing the difference between fake news and real news.

    Parameters:
    ============
    train_ds_char_count: pandas.DataFrame
              The train dataframe containing all the initial informations and
              the number of characters for each single tweet.

    val_ds_char_count: pandas.DataFrame
              The validation dataframe containing all the initial informations and
              the number of characters for each single tweet.
    
    test_ds_char_count: pandas.DataFrame
              The test dataframe containing all the initial informations amd
              the number of characters for each single tweet.
    '''                             
    print("\nReal news length (average chars):"
          "training {0:.1f}, validation {1:.1f}, test {2:.1f}".format(
        train_ds_char_count[train_ds_char_count['label']=='real']['char_count'].mean(),
        val_ds_char_count[val_ds_char_count['label']=='real']['char_count'].mean(),
        test_ds_char_count[test_ds_char_count['label']=='real']['char_count'].mean()))

    print("Fake news length (average chars):" 
          "training {0:.1f}, validation {1:.1f}, test {2:.1f}".format(
        train_ds_char_count[train_ds_char_count['label']=='fake']['char_count'].mean(),
        val_ds_char_count[val_ds_char_count['label']=='fake']['char_count'].mean(),
        test_ds_char_count[test_ds_char_count['label']=='fake']['char_count'].mean()))



def plotting_char_count(train_ds_char_count: pd.DataFrame,
                        val_ds_char_count: pd.DataFrame,
                        test_ds_char_count: pd.DataFrame) -> None:
    '''
    This function plots the average number of characters for each dataframe
    comparing the difference between fake news and real news.

    Parameters:
    ============
    train_ds_with_word_count: pandas.DataFrame
              The train dataframe containing all the initial informations and
              the number of characters for each single tweet.

    val_ds_with_word_count: pandas.DataFrame
              The validation dataframe containing all the initial informations and
              the number of characters for each single tweet.
    
    test_ds_with_word_count: pandas.DataFrame
              The test dataframe containing all the initial informations amd
              the number of characters for each single tweet.
    '''  
    # PLOTTING CHAR-COUNT
    sns.set(font_scale=1.4)
    complete_ds = pd.concat([train_ds, val_ds, test_ds], ignore_index = True)

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    
    words=complete_ds[complete_ds['label']=='real']['char_count']
    ax1.hist(words,color='red', range = (0, 400))
    ax1.set_title('Real news COVID19')
    
    words=complete_ds[complete_ds['label']=='fake']['char_count']
    ax2.hist(words,color='green', range = (0, 400))
    ax2.set_title('Fake news COVID19')
    
    fig.suptitle('Chars per tweet')
    plt.show()


def explore(input_folder):
    '''
    This function executes all the previous functions in order to 
    visualize, describe, analyze and explore the data.
    Parameters:
    ============
    input_folder: string
              The name of the folder where the three csv files with data are contained.
    '''
    df_train, df_val, df_test = read_data(input_folder)

    info_data(df_train, df_val, df_test)
    plot_label_distribution(df_train, df_val, df_test)

    df_all_word_count = word_count_twitter(df_train, df_val, df_test)
    df_train_word_count, df_val_word_count, df_test_word_count = df_all_word_count
    word_count_printer(df_train_word_count, 
                        df_val_word_count,
                        df_test_word_count)

    plotting_word_count(df_train_word_count, 
                        df_val_word_count,
                        df_test_word_count)

    df_all_char_count = char_count_twitter(df_train_word_count, 
                                            df_val_word_count,
                                            df_test_word_count)
    df_train_char_count, df_val_char_count, df_test_char_count = df_all_char_count
    char_count_printer(df_train_char_count,
                       df_val_char_count,
                       df_test_char_count)
    plotting_char_count(df_train_char_count,
                       df_val_char_count,
                       df_test_char_count)                

def main():
    config_parse = configparser.ConfigParser()
    configuration = config_parse.read(sys.argv[1])
    
    input_folder = config_parse.get('INPUT_OUTPUT', 'input_folder')
    explore(input_folder = input_folder)

if __name__ == '__main__':
    main()