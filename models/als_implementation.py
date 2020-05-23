import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from tqdm import tqdm


def als_algorithm(dataset, regularization=0.1, alpha_parameter=40, iterations=10, factors=20, random_state=0):
    """Implements implicit ALS (Alternating Least Squares) algorithm for collaborative filtering.

    Parameters
    ----------
    dataset:
        Training dataset.
    regularization:
        Regularization parameter (used for creating regularization term).
    alpha_parameter:
        Alpha parameter (used for creating confidence matrix).
    iterations:
        Number of iterations.
    factors:
        Number of user and item factors.
    random_state:
        Random state for feature vector initialization.

    Returns
    -------
        user_matrix, item_matrix.T: Matrices with feature vectors for users and items.
    """
    # Create confidence matrix
    confidence_matrix = alpha_parameter * dataset
    # To allow the matrix to stay sparse, I will add one later when each row is taken and converted to dense.
    num_users, num_items = confidence_matrix.shape[0], confidence_matrix.shape[1]

    # Set random seed for feature vector initialization
    random_state = np.random.RandomState(random_state)

    # Create sparse matrices for users and items with given rank and initialize them with random_state
    user_matrix = sparse.csr_matrix(random_state.normal(size=(num_users, factors)))  # Matrix: m x factors
    item_matrix = sparse.csr_matrix(random_state.normal(size=(num_items, factors)))  # Matrix: n x factors

    # Create sparse eye matrices for users and items
    user_eye = sparse.eye(num_users)
    item_eye = sparse.eye(num_items)

    # Create regularization term: Lambda * I
    lambda_eye = regularization * sparse.eye(factors)

    # Begin iterations (solving user given fixed items and items given fixed users)
    for _ in tqdm(range(iterations)):
        userTuser = user_matrix.T.dot(user_matrix)
        itemTitem = item_matrix.T.dot(item_matrix)

        # Solve users based on fixed items
        for user in tqdm(range(num_users)):
            # Select confidence matrix row for given user
            confidence_row = confidence_matrix[user, :].toarray()

            # Create binarized preference vector
            preference_vector = confidence_row.copy()
            preference_vector[preference_vector != 0] = 1

            # Increment user index so every user-item is given minimal confidence
            confidence_row += 1

            # Calculate Cu - I term
            CuI = sparse.diags(confidence_row, [0])

            # Calculate itemT*(Cu - I)*item  term
            itemTCuIitem = item_matrix.T.dot(CuI).dot(item_matrix)

            # Calculate itemT*Cu*Pu term
            itemTCuPu = item_matrix.T.dot(CuI + item_eye).dot(preference_vector.T)

            # Solve for given user
            user_matrix[user] = spsolve(itemTitem + itemTCuIitem + lambda_eye, itemTCuPu)

        # Solve items based on fixed users
        for item in tqdm(range(num_items)):
            # Select confidence matrix row for given item
            confidence_row = confidence_matrix[:, item].T.toarray()

            # Create binarized preference vector
            preference_vector = confidence_row.copy()
            preference_vector[preference_vector != 0] = 1

            # Increment user index so every item-user is given minimal confidence
            confidence_row += 1

            # Calculate Ci - I term
            CiI = sparse.diags(confidence_row, [0])

            # Calculate userT*(Ci - I)*user term
            userTCiIuser = user_matrix.T.dot(CiI).dot(user_matrix)

            # Calculate userT*Ci*Pi term
            userTCiPi = user_matrix.T.dot(CiI + user_eye).dot(preference_vector.T)

            # Solve for given item
            item_matrix[item] = spsolve(userTuser + userTCiIuser + lambda_eye, userTCiPi)

    return user_matrix, item_matrix.T
