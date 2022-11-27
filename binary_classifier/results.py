from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

'''
def prediction(model, X_test, neural_network = False):
    y_predict = model.predict(X_test)
    if neural_network is True:
        y_prob = y_predict
        y_predict[y_predict > 0.5] = 1
        y_predict[y_predict <= 0.5] = 0
    else:
        y_prob = model.predict_proba(X_test)[:,1]
    return y_predict, y_prob
'''

def visualize_results(y_test, y_predict, y_prob,
                      labels, title, 
                      history = None,
                      folder_path = None):
    print(classification_report(y_test,y_predict))
    cm = confusion_matrix(y_test, y_predict)
    print('Confusion Matrix:', cm)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)

    cm_plot_path = folder_path / (title + '.svg')
    make_confusion_matrix(cm, labels, title = title, filepath=cm_plot_path)
    
    if history is not None:
        history_plot_path = folder_path / 'history.svg'
        plot_history(history, filepath=history_plot_path)


def make_confusion_matrix(cf,
                          group_names=None,
                          count=True,
                          percent=True,
                          figsize=None,
                          cmap='Blues',
                          title=None,
                          filepath=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    count:         If True, show the raw number in the confusion matrix. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    filepath:      File path where to save the figure. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]
    sns.set(font_scale=1.2)

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])

    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')


    # MAKE THE HEATMAP VISUALIZATION
    fig, ax= plt.subplots(1, 1, figsize = figsize)
    sns.heatmap(cf, annot=box_labels, fmt='', ax=ax, cmap = cmap);  #annot=True to annotate cells, ftm='g' to disable scientific notation

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



def plot_history(history, filepath=None):
    sns.set(font_scale=1.4)

    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    x = range(1, len(acc) + 1)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12, 10))
    ax1.plot(x, acc, 'b', label='Training acc')
    ax1.plot(x, val_acc, 'r', label='Validation acc')
    ax1.set_title('Training and validation accuracy')
    ax1.legend()

    ax2.plot(x, loss, 'b', label='Training loss')
    ax2.plot(x, val_loss, 'r', label='Validation loss')
    ax2.set_title('Training and validation loss')
    ax2.legend()
    
    plt.show()

    if filepath is not None:
        fig.savefig(filepath)