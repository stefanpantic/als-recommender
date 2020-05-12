from sklearn import metrics


def auc_score(test, predictions):
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def mean_auc(train_set, altered_users, predictions, test_set):
    pass
