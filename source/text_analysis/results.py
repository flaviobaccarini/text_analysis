'''
RESULTS MODULE
===============
In this module there are functions to visualize and plotting 
the test of results (both for the neural network or the machine learning model)
'''
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path

def visualize_results(y_test: np.ndarray,
                      y_predict: np.ndarray,
                      y_prob: np.ndarray,
                      classes: ArrayLike,
                      name_model: str, 
                      metrics: list[str] = None,
                      history: dict = None,
                      folder_path: str = None) -> None:
    '''
    Function to visualize the results after the predict process.
    It takes as input the test y, the predicted y and the probability
    for y (probability to belong to a specific class).
    The function computes the confusion matrix, the roc curve and it prints 
    out the classification report. It also plots the confusion matrix thank to
    the make_confusion_matrix function and in the case of the neural network
    it plots also the training history.
    If the user provides a folder path, the plots are saved inside this folder.

    Parameters:
    ===========
    y_test: 1-D np.ndarray
            The real y value.
    
    y_predict: 1-D np.ndarray
               The y value predicted by the model.

    y_prob: 1-D np.ndarray
            The probability for each single y to belong
            to a specific class.
    
    classes: 1-D array-like
            This sequence represents the original unique 
            class names for the labels.

    name_model: str
                The name of the model.
    
    metrics: list[str] default: None
             Sequence of metrics used for the neural network. Default is None,
             necessary for neural network history.

    history: dict default: None
             If provided, dictionary that contains all
             the information about the training of the
             neural network model.
    
    folder_path: Path-like or str default: None
                 If provided, the path where to save
                 the plots generated from this function.
                 Default is None.
    '''
    print(classification_report(y_test,y_predict))
    cm = confusion_matrix(y_test, y_predict)
    print('Confusion Matrix:', cm)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)

    if folder_path is not None:
        if type(folder_path) == 'str':
            folder_path = Path(folder_path)
        cm_plot_path = folder_path / (name_model + '.svg')
        make_confusion_matrix(cm, classes, title = name_model, filepath=cm_plot_path)
    else:
        make_confusion_matrix(cm, classes, title = name_model)
    
    if history is not None:
        if folder_path is not None:
            history_plot_path = folder_path / 'history.svg'
            plot_history(history, metrics_name = metrics, filepath = history_plot_path)
        else:
            plot_history(history, metrics_name = metrics)      


def make_confusion_matrix(cm: np.ndarray,
                          group_names: ArrayLike,
                          count: bool=True,
                          percent: bool=True,
                          figsize: tuple=None,
                          cmap: str='Blues',
                          title: str=None,
                          filepath: str=None) -> None:
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix (cm)
    using a Seaborn heatmap visualization. 
    
    Parameters:
    ===========
    cm: np.ndarray
        confusion matrix to be passed in

    group_names: 1-D array-like
                 List of strings that represent the labels row by row
                 to be shown in each square.

    count: bool default: True
           If True, show the raw number in the confusion matrix.
           Default is True.

    percent: bool default: True
             If True, show the percent number in the confusion matrix.
             Default is True.

    figsize: tuple[number] default: None
             Tuple representing the figure size.
             Default will be the matplotlib rcParams value.

    cmap: str default: 'Blues'
          Colormap of the values displayed from matplotlib.pyplot.cm.
          Default is 'Blues'
          See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title: str default: None
           Title for the heatmap. Default is None.
    
    filepath: Path-like or str default: None
              File path where to save the figure. Default is None.
    '''
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cm.size)]
    sns.set(font_scale=1.2)

    if len(group_names)==cm.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cm.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value 
                                                        in cm.flatten()/np.sum(cm)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in 
                                zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cm.shape[0],cm.shape[1])

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    # MAKE THE HEATMAP VISUALIZATION
    fig, ax= plt.subplots(1, 1, figsize = figsize)
    sns.heatmap(cm, annot=box_labels, fmt='', ax=ax, cmap = cmap);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    title_plot = title + " confusion matrix"
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(group_names)
    ax.yaxis.set_ticklabels(group_names)
    ax.set_title(title_plot)
    plt.show()

    if filepath is not None:
        fig.savefig(filepath)



def plot_history(history: dict,
                 metrics_name: list[str],
                 filepath: str=None) -> None:
    '''
    This function will make a plot of the history for the training of 
    the neural network. In particular, losses and metrics will be 
    plot.
    
    Parametes:
    ==========
    history: dict
             Dictionary that contains values for the 
             losses and metrics used during the neural network train.
    
    metrics_name: list[str]
    		      Sequence of metric names used for training the model.
    
    filepath: Path-like or str default: None
              File path where to save the figure. Default is None.
    '''
    sns.set(font_scale=1.2)

    loss = history['loss']
    val_loss = history['val_loss']

    fig, ax = plt.subplots(1, len(metrics_name) + 1, figsize=(12, 10))
    x = range(1, len(loss) + 1)
    ax[0].plot(x, loss, 'b', label='Training loss')
    ax[0].plot(x, val_loss, 'r', label='Validation loss')
    ax[0].set_title('Training and validation loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
        
    for index, metric in enumerate(metrics_name):
        train_metric = history[metric]
        valid_metric = history['val_'+metric]
        ax[index+1].plot(x, train_metric, 'b', label='train_'+metric)
        ax[index+1].plot(x, valid_metric, 'r', label='val_'+metric)
        ax[index+1].set_title(metric)
        ax[index+1].set_xlabel('Epochs')
        ax[index+1].set_ylabel('Metric score')
        ax[index+1].legend()
    
    plt.show()

    if filepath is not None:
        fig.savefig(filepath)
