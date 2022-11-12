
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score


def prediction(model, X_test):
    y_predict = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    return y_predict, y_prob

def visualize_results(y_test, y_predict, y_prob):
    print(classification_report(y_test,y_predict))
    print('Confusion Matrix:',confusion_matrix(y_test, y_predict))
    
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)
