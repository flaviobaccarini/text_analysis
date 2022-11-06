from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def read_data(name_folder = 'datasets'):
    datasets_path = Path(name_folder)
    csv_paths = list(datasets_path.glob('**/*.csv'))
    csv_paths_stem = [x.stem + x.suffix for x in csv_paths]
    csv_paths_stem_sorted = sorted(csv_paths_stem)
    train_ds = pd.read_csv(datasets_path / csv_paths_stem_sorted[0])
    val_ds = pd.read_csv(datasets_path / csv_paths_stem_sorted[1])
    test_ds = pd.read_csv(datasets_path / csv_paths_stem_sorted[2])
    return train_ds, val_ds, test_ds


def info_data(train_ds, val_ds, test_ds):
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
    

def plot_label_distribution(train_ds, val_ds, test_ds):

    sns.set(font_scale=1.4)
    fig, ax = plt.subplots()
    #train_ds['label'].value_counts().plot(kind='bar', figsize=(7, 6), rot=0, ax = ax)
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
    ax2.set_title("Count of Real/Fake News for test set", y=1.02);

    plt.show()


def word_count_twitter(train_ds, val_ds, test_ds):
    train_ds_with_word_count = train_ds.copy()
    val_ds_with_word_count = val_ds.copy()
    test_ds_with_word_count = test_ds.copy()

    train_ds_with_word_count['word_count'] = train_ds_with_word_count['tweet'].apply(lambda x: len(str(x).split()))
    val_ds_with_word_count['word_count'] = val_ds_with_word_count['tweet'].apply(lambda x: len(str(x).split()))
    test_ds_with_word_count['word_count'] = test_ds_with_word_count['tweet'].apply(lambda x: len(str(x).split()))

    return train_ds_with_word_count, val_ds_with_word_count, test_ds_with_word_count

def word_count_printer(
                        train_ds_with_word_count,
                        val_ds_with_word_count,
                        test_ds_with_word_count
                        ):
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


def plotting_word_count(train_ds, val_ds, test_ds):

    # PLOTTING WORD-COUNT
    sns.set(font_scale=1.4)
    complete_ds = pd.concat([train_ds, val_ds, test_ds], ignore_index = True)

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    
    words=complete_ds[complete_ds['label']=='real']['word_count']
    ax1.hist(words,color='red', range = (0, 50))
    ax1.set_title('Real news COVID19')
    
    words=complete_ds[complete_ds['label']=='fake']['word_count']
    ax2.hist(words,color='green', range = (0, 50))
    ax2.set_title('Fake news COVID19')
    
    fig.suptitle('Words per tweet')
    plt.show()

def char_count_twitter(train_ds, val_ds, test_ds):
    train_ds_char_count = train_ds.copy()
    val_ds_char_count = val_ds.copy()
    test_ds_char_count = test_ds.copy()

    train_ds_char_count['char_count'] = train_ds_char_count['tweet'].apply(lambda x: len(str(x)))
    val_ds_char_count['char_count'] = val_ds_char_count['tweet'].apply(lambda x: len(str(x)))
    test_ds_char_count['char_count'] = test_ds_char_count['tweet'].apply(lambda x: len(str(x)))

    return train_ds_char_count, val_ds_char_count, test_ds_char_count

def char_count_printer(
                        train_ds_char_count,
                        val_ds_char_count,
                        test_ds_char_count
                        ):
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



def plotting_char_count(train_ds, val_ds, test_ds):

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

'''
train, val, test = read_data()
info_data(train, val, test)
plot_label_distribution(train, val, test)

train_word_count, val_word_count, test_word_count = word_count_twitter(train, val, test)
word_count_printer(train_word_count, val_word_count, test_word_count)
plotting_word_count(train_word_count, val_word_count, test_word_count)

all_ds = char_count_twitter(train_word_count, val_word_count, test_word_count)
train_word_char_count_ds, val_word_char_count_ds, test_word_char_count_ds = all_ds
char_count_printer(train_word_char_count_ds,
                   val_word_char_count_ds,
                   test_word_char_count_ds)
plotting_char_count(train_word_char_count_ds,
                    val_word_char_count_ds,
                    test_word_char_count_ds)
'''