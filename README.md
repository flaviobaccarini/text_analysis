# Binary text classification

## Introduction
The aim of this project is to build a little library for text analysis. The specific library's aim is to make a pre-analysis of the data, preprocessing of the data and then plotting of the results of the generated models by the user. The model selection, training and evaluation is up to the user. Starting from this library it is possibile to build different machine learning or deep learning models in order to make a binary classification of the text. At this moment three different datasets have been analyzed. The first is a dataset containing COVID-19 tweets: the aim is to classifiy if the tweet corresponds to a real or fake news. The second one is a spam/ham classification of messages. The last one is a classification of disaster tweets: similar to COVID-19 tweet dataset, the aim is a classification for real or fake news about natural disasters. 
* In the folder [library](https://github.com/flaviobaccarini/text_analysis/tree/main/text_analysis) the library can be found with all the test functions.
* In the folder [datasets](https://github.com/flaviobaccarini/text_analysis/tree/main/datasets) the user can find the three datasets that the user can analyze.
* In the folder [scripts](https://github.com/flaviobaccarini/text_analysis/tree/main/scripts) the user can find the scripts used for the text analysis.
* In the folder [tutorials](https://github.com/flaviobaccarini/text_analysis/tree/main/tutorials) the user can find two tutorials about the text analysis with this library.

Two main models have been evaluated for all the datasets: logistic regressor with a Word2Vec model (to vectorize the text) and a Bidirectional LSTM neural network.

## Installing the library
Firstly it is necessary to install the library. So first of all, clone the repository with:\
`git clone https://github.com/flaviobaccarini/text_analysis.git`\
then\
`cd text_analysis/source`\

In order to install the library some other libraries are necessary, so we can install them with:\
`pip install setuptools twine wheel`\

Now we can run the commands for installing the library:\
```
python setup.py bdist_wheel
pip install ./dist/text_analysis-0.1.0-py3-none-any.whl
```

Now the library should be installed.  
Running:\
`python setup.py pytest`\
will execute all tests stored in the ‘tests’ folder.

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
* The [explore](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/explore.py) file corresponds to the pre-classification file: some statistical data analysis to understand which kind of data the user is facing.  
* The [preprocess](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/preprocess.py) file corresponds to the data preprocessing (in particular cleaning of the text).
* In the [train_lr](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/train_lr.py) file the user can find the training of a logistic regressor model for binary text classification.
* In the [train_nn](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/train_nn.py) file the user can find the training of a neural network (Bidirectional LSTM) built with the tensorflow library for binary text classification. The model is compiled using a binary cross entropy loss and an accuracy metrics.
* The [evaluate_lr](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/evaluate_lr.py) file contains the evaluation/test of the trained logistic regressor model. 
* The [evaluate_nn](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/evaluate_nn.py) file contains the evaluation/test of the trained neural network (Bidirectional LSTM) model. 
* Each single script file has to be executed with a config file (see [config.ini](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/config.ini) and [configuration](#configuration-file) section).

For example, if the user wants to train and test a logistic regressor model, the commands needed to be run are the following:\
(OPTIONAL, BUT RECCOMMENDED) `python explore.py config.ini`
```
python preprocess.py config.ini
python train_lr.py config.ini
python evaluate_lr.py config.ini
```

If the user wants to train and test a neural network:\
(OPTIONAL, BUT RECCOMMENDED) `python explore.py config.ini`
```
python preprocess.py config.ini
python train_nn.py config.ini
python evaluate_nn.py config.ini
```

After the execution of `preprocess.py` a new folder will be created. In this new folder there will be all the preprocessed datasets. 
After the execution of `train_lr.py` or `train_nn.py` another folder will be created. The trained models will be contained in this new folder. `train_lr.py` will save both the logistic regressor and the Word2Vec model (the Word2Vec model is used to convert the text to vector numbers), while `train_nn.py` will just save the neural network model. After the execution of `evaluate_lr.py` or `evaluate_nn.py` a new folder will be created. The plots generated by the execution will be saved in this new folder.


## Configuration file
The configuration file [config.ini](https://github.com/flaviobaccarini/text_analysis/blob/main/scripts/config.ini) is composed by three different sections:
1) ANALYSIS 
2) PREPROCESS 
3) PARAMETERS TRAIN 
In the first section (ANALYSIS) the only variable specifies the name of the sub-folder (inside the [datasets](https://github.com/flaviobaccarini/text_analysis/tree/main/datasets)) where the data are stored. 
It is assumed that the the folder structure is someting like this:
```
text_analysis
│   README.md
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
This means that the csv files need to be stored inside a sub-folder called as the analysis. The variable (folder_name) inside the ANALYSIS section of the configuration file has to be initialized with the name of the analysis the user wants to perform.

Then, the second section of the configuration file is called PREPROCCESED. In this section the fraction numbers for splitting the dataset (if necessary) are initializated and also the seed in order to have reproducibility.

The third section is called PARAMETERS_TRAIN and it contains the train parameters for the neural network and the logistic regressor.

## Tutorials
In the [tutorials](https://github.com/flaviobaccarini/text_analysis/tree/main/tutorials) the user can find two tutorials about the text analysis with this library. In one tutorial the train and test of a logistic regressor on spam dataset is shown, instead in the other tutorial the train and test of a Bidirectional LSTM on covid dataset is presented.