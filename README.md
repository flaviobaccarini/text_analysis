# Binary text classification

## Introduction
The aim of this project is to build a little library for text analysis: in particular binary text classifcation. The specific library's aim is to make a pre-analysis of the data, preprocessing of the data and then plotting of the results of the generated models by the user. The model selection, training and evaluation is up to the user. Starting from this library it is possibile to build different machine learning or deep learning models in order to make a binary classification of the text. At this moment three different datasets have been analyzed. The first is a dataset containing COVID-19 tweets: the aim is to classifiy if the tweet corresponds to a real or fake news. The second one is a spam/ham classification of messages. The last one is a classification of disaster tweets: similar to COVID-19 tweet dataset, the aim is a classification for real or fake news about natural disasters. 
* In the folder [library](https://github.com/flaviobaccarini/text_analysis/tree/main/source) the library can be found with all the test functions.
* In the folder [datasets](https://github.com/flaviobaccarini/text_analysis/tree/main/datasets) the user can find the three datasets that have been analyzed until now.
* In the folder [scripts](https://github.com/flaviobaccarini/text_analysis/tree/main/scripts) the user can find the scripts used for the text analysis.
* In the folder [tutorials](https://github.com/flaviobaccarini/text_analysis/tree/main/tutorials) the user can find two tutorials about the text analysis with this library.

Two main models have been evaluated for all the datasets: logistic regressor with a Word2Vec model (to vectorize the text) and a Bidirectional LSTM neural network.

## Installing the library
Firstly it is necessary to install the library. So first of all, clone the repository with:
```
git clone https://github.com/flaviobaccarini/text_analysis.git
cd text_analysis/source
```

In order to install the library some other libraries are necessary, so we can install them with:\
`pip install setuptools twine wheel`

Now we can run the commands for installing the library:
```
python3 setup.py bdist_wheel
pip install ./dist/text_analysis-0.1.0-py3-none-any.whl
```

Now the library should be installed.  
Running:\
`pytest tests`\
will execute all test functions inside the `tests` folder.
The library was built and tested with python version `3.10.6`.

## Library structure
The aim of the library is to make an initial exploration data analysis in order to better understand the data, then the library is aimed to preprocess the data: both cleaning and vectorization of the text data. The library is also composed by functions for plotting the results coming from the evaluation of the models. The selection, training and evaluation of the machine or deep learning model is up to the user. 
The library is composed by different modules:
* [read_and_write](https://github.com/flaviobaccarini/text_analysis/blob/main/source/text_analysis/read_write_data.py) takes care of input output: in particular, in this module there are functions for reading data (in csv format) and writing the preprocessed data (in csv format) in a new folder.
* [split_dataframe](https://github.com/flaviobaccarini/text_analysis/blob/main/source/text_analysis/split_dataframe.py) contains all the functions useful for the splitting of the initial dataset(s), so that to obtain three different datasets (train, validation and test).
* In [preanalysis](https://github.com/flaviobaccarini/text_analysis/blob/main/source/text_analysis/preanalysis.py) the user can find some functions for making an initial data exploration pre-analysis. 
* [cleantext](https://github.com/flaviobaccarini/text_analysis/blob/main/source/text_analysis/cleantext.py) is the module containing all the functions for cleaning the text data.
* The aim of [vectorize_data](https://github.com/flaviobaccarini/text_analysis/blob/main/source/text_analysis/vectorize_data.py) module is the text data vectorization after the cleaning. The vectorization is made with a Word2Vec model for machine larning models, or tensorflow TextVectorization layer for neural network models.
* [results](https://github.com/flaviobaccarini/text_analysis/blob/main/source/text_analysis/results.py) contains functions for plotting the results obtained by the trained models.

## Project structure
Inside the [scripts](https://github.com/flaviobaccarini/text_analysis/tree/main/scripts) folder the user can find all the useful scripts in order to make a binary text classification. The script files are the following:
* [explore](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/explore.py) corresponds to the pre-analysis script: some statistical data analysis to understand which kind of data the user is facing.  
* [preprocess](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/preprocess.py) corresponds to the data preprocessing (in particular cleaning of the text).
* In [train_lr](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/train_lr.py) the user can find the training of a logistic regressor model for binary text classification.
* In [train_nn](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/train_nn.py) the user can find the training of a neural network (Bidirectional LSTM) built with the tensorflow library for binary text classification. The model is compiled using a binary cross entropy loss and an accuracy metric.
* [evaluate_lr](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/evaluate_lr.py) contains the evaluation/test of the trained logistic regressor model. 
* [evaluate_nn](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/evaluate_nn.py) contains the evaluation/test of the trained neural network (Bidirectional LSTM) model. 
* Each single script file has to be executed with a config file (see [config.ini](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/config.ini) and [configuration](#configuration-file) section).

For example, if the user wants to train and test a logistic regressor model, the commands needed to be run are the following:\
(OPTIONAL, BUT RECCOMMENDED) `python3 explore.py config.ini`
```
python3 preprocess.py config.ini
python3 train_lr.py config.ini
python3 evaluate_lr.py config.ini
```

If the user wants to train and test a neural network:\
(OPTIONAL, BUT RECCOMMENDED) `python3 explore.py config.ini`
```
python3 preprocess.py config.ini
python3 train_nn.py config.ini
python3 evaluate_nn.py config.ini
```

After the execution of `preprocess.py` a new folder will be created (see [preproccessed_datasets](https://github.com/flaviobaccarini/text_analysis/tree/main/scripts/preprocessed_datasets)). In this new folder there will be all the preprocessed datasets. 
After the execution of `train_lr.py` or `train_nn.py` another folder will be created (see [checkpoint](https://github.com/flaviobaccarini/text_analysis/tree/main/scripts/checkpoint)). The trained models will be contained in this new folder. `train_lr.py` will save both the logistic regressor and the Word2Vec model (the Word2Vec model is used to convert the text to vector numbers), while `train_nn.py` will just save the neural network model. After the execution of `evaluate_lr.py` or `evaluate_nn.py` a new folder will be created (see [plots](https://github.com/flaviobaccarini/text_analysis/tree/main/scripts/plots)). The plots generated by the execution will be saved in this new folder.


## Configuration file
The configuration file [config.ini](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/config.ini) is composed by three different sections:
1) ANALYSIS 
2) PREPROCESS 
3) PARAMETERS TRAIN \
In the first section (ANALYSIS) the variable folder_name defined in this section specifies the name of the sub-folder (inside the [datasets](https://github.com/flaviobaccarini/text_analysis/tree/main/datasets) folder) where the data are stored. 
It is assumed that the folder structure is something like this:
```
text_analysis
│   README.md
│   scripts
│   datasets
│   source
│   ...
│
└───datasets
   └───spam
   │   │   alldata.csv
   │   │...
   │   
   └───covid       
   │   │    train.csv
   │   │    validation.csv
   │   │    test.csv
   │
   └───disaster
       │   train.csv
       │   test.csv
       │...

```
This means that the csv files (containing the data) need to be stored inside a sub-folder called as the analysis (for more details on the data format see the [prerequisites for the data](#data-prerequisites) section). The variable called folder_name inside the ANALYSIS section of the configuration file has to be initialized with the name of the analysis the user wants to perform (at this moment 3 possible options: "covid", "spam", "disaster").
Then we find, always in the ANALYSIS section, two other variables:
1) text_column_name: it is the text column name inside the original dataframe (for example, for spam dataset it is equal to original_message)
2) label_column_name: it is the label column name inside the original dataframe (for example, for spam dataset it is equal to spam)

Then, the second section of the configuration file is called PREPROCCESED. In this section the fraction numbers for splitting the dataset (if necessary) are initializated. A random seed is also set in order to have reproducibility.

The third section is called PARAMETERS_TRAIN and it contains the train parameters for the neural network and the logistic regressor:
* embedding_vector_size: is the size of the vector after the text vectorization
* batch_size: is the size of the batch for the neural network training
* epochs: is the maximum number of epochs for the neural network training
* learning_rate: is the learning rate for the neural network training
* min_count_words_w2v: is the minimum number of occurencies for each word, so that the word can be counted in the vocabolary for the Word2Vec model

## Data prerequisites
Here some basic prerequisites for the data:
* all the data need to be stored in csv files
* there could be three different options:
    1) one single csv file: it is assumed that alle data are stored inside this single file
    2) two different csv files: it is assumed that one file is for training part and the other is for testing part. The train csv file has to contain the 'train' word inside the file name
    3) three different csv files: it is assumed that one file is for training part, one is for validation part and the last one is for the testing part. The train csv file has to contain the 'train' word inside the file name, while the validation csv file has to contain the 'val' word in the file name, while the test csv file has to contain the 'test' word inside the file name. 
    
Example:
```
datasets
└───spam
│   │   the_file_name_could_be_everything.csv
│   
└───covid       
│   │    train_covid_tweet.csv
│   │    val_covid_tweet.csv
│   │    test_covid_tweet.csv
│
└───disaster
    │   train_distater_tweet.csv
    │   random_name_disaster_tweet.csv

```
The `spam` dataset presents only one single file: all the data are in `the_file_name_could_be_everything.csv`. \
The `covid` dataset presents three different csv files: `train_covid_tweet.csv` corresponds to the train data, `val_covid_tweet.csv` corresponds to the validation data and `test_covid_tweet.csv` corresponds to the test data. \
The `disaster` dataset presents two different csv files: `train_disaster_tweet.csv` corresponds to the train data, while `random_name_disaster_tweet.csv` (no request on this file name) will be assumed to correspod to the test data. 

## Tutorials
In the [tutorials](https://github.com/flaviobaccarini/text_analysis/tree/main/tutorials) the user can find two tutorials about text analysis with this library. In one tutorial the train and test of a logistic regressor on spam dataset is shown (see [tutorial logistic regressor](https://github.com/flaviobaccarini/text_analysis/blob/main/tutorials/Tutorial%20Spam%20Logistic%20Regressor.ipynb)), instead in the other tutorial the train and test of a Bidirectional LSTM on covid dataset is presented (see [tutorial bidirectional LSTM](https://github.com/flaviobaccarini/text_analysis/blob/main/tutorials/Tutorial%20Covid%20Bidirectional%20LSTM.ipynb)).
