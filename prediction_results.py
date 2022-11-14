
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from plotting_result import make_confusion_matrix, plot_history

def prediction(model, X_test, neural_network = False):
    y_predict = model.predict(X_test)
    if neural_network is True:
        y_prob = y_predict
        y_predict[y_predict > 0.5] = 1
        y_predict[y_predict <= 0.5] = 0
    else:
        y_prob = model.predict_proba(X_test)[:,1]
    return y_predict, y_prob

def visualize_results(y_test, y_predict, y_prob, labels, title, history = None,
                      folder_path = None):
    print(classification_report(y_test,y_predict))
    cm = confusion_matrix(y_test, y_predict)
    print('Confusion Matrix:', cm)
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)

    if history is not None:
        history_plot_path = folder_path / 'history.svg'
        plot_history(history, filepath=history_plot_path)

    cm_plot_path = folder_path / (title + '.svg')
    make_confusion_matrix(cm, labels, title = title, filepath=cm_plot_path)
    



