from sklearn import metrics
import numpy as np


def auc_score(test, predictions):
    """Calculates AUC score.

    Parameters
    ----------
    test:
        Test set.
    predictions:
        Recommender predictions.

    Returns
    -------
        AUC score for given test and prediction sets.
    """
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)

    return metrics.auc(fpr, tpr)


def calculate_mean_auc(train_set, test_users, predictions, test_set):
    """Calculates mean AUC scores for test users and popular items.

    Parameters
    ----------
    train_set:
        Training set.
    test_users:
        Users that had an item removed from the training set.
    predictions:
        Recommendations for test users.
    test_set:
        Test set.

    Returns
    -------
        Mean AUC scores.
    """
    # Empty list for AUC of each user that had an item removed from training set
    auc = []
    # Empty list for popular AUC scores
    popularity_auc = []
    # Find most popular items
    popular_items = np.array(test_set.sum(axis=0)).reshape(-1)
    item_vecs = predictions[1]

    for user in test_users:
        # Get the train set row for user
        train_row = train_set[user, :].toarray().reshape(-1)
        # Find the place where user didn't interact with item
        zero_inds = np.where(train_row == 0)
        # Get predicted values from user/item vectors
        user_vec = predictions[0][user, :]
        # Get only original zero items
        pred = user_vec.dot(item_vecs).toarray()[0, zero_inds].reshape(-1)
        # Get all ratings from predictions for this user that originally had no interaction
        test = test_set[user, :].toarray()[0, zero_inds].reshape(-1)
        # Get item popularity for chosen items
        popularity = popular_items[zero_inds]
        # Calculate AUC scores
        auc.append(auc_score(predictions=pred, test=test))
        popularity_auc.append(auc_score(predictions=popularity, test=test))

    # Calculate mean AUC scores
    mean_auc = np.mean(auc)
    mean_popular_auc = np.mean(popularity_auc)

    return mean_auc, mean_popular_auc
