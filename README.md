# Binary text classification

## Introduction
The aim of this project is to build a little library for some text analysis. In particular, starting from this library it is possibile to build different machine learning or deep learning models in order to make a binary classification of the text. At this moment three different datasets have been analyzed. The first is a dataset containing COVID-19 tweets: the aim is to classifiy if the tweet corresponds to a real or fake news. The second one is a spam/ham classification of messages. The last one is a classification of disaster tweets: similar to COVID-19 tweet dataset, the aim is a classification for real or fake news about natural disasters. 
* In the folder [library](https://github.com/flaviobaccarini/text_analysis/tree/main/text_analysis) the library can be found with all the test functions.
* In the folder [datasets](https://github.com/flaviobaccarini/text_analysis/tree/main/datasets) the user can find the three datasets that the user can analyze.
* In the folder [scripts](https://github.com/flaviobaccarini/text_analysis/tree/main/scripts) the user can find the scripts used for the text analysis.
* In the folder [tutorials](https://github.com/flaviobaccarini/text_analysis/tree/main/tutorials) the user can find two tutorials about the text analysis with this library.

Two main models have been evaluated for all the datasets: logistic regressor and a Bidirectional LSTM neural network.

## Installing the library
Firstly it is necessary to install the library. So first of all, clone the repository with: 
`git clone https://github.com/flaviobaccarini/text_analysis.git` 
then
`cd text_analysis/source`

In order to install the library some other libraries are necessary, so we can install them with:
`pip install setuptools twine wheel`

Now we can run the commands for installing the library:
`python setup.py bdist_wheel`
`pip install ./dist/text_analysis-0.1.0-py3-none-any.whl`

Now the library should be installed.
Running:
`python setup.py pytest`
will execute all tests stored in the ‘tests’ folder.
