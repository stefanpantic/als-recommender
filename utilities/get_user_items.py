import numpy as np


def get_user_items(user_id, train_matrix, user_list, item_list):
    """Get watched movies for given user.

    Parameters
    ----------
    user_id:
        Id of the user.
    train_matrix:
        Train set.
    user_list:
        List of all users.
    item_list:
        List of all movies.

    Returns
    -------
        List of movies rated by given user.
    """
    user_ind = np.where(user_list == user_id)[0][0]
    items_ind = train_matrix[user_ind, :].nonzero()[1]

    return item_list[items_ind]
