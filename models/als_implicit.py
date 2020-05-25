from implicit.als import AlternatingLeastSquares


def get_implicit_als(dataset, regularization, alpha_parameter, iterations, factors, random_state):
    """Gets results from ALS algorithm implementation from implicit library.

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
        user_matrix, item_matrix: Matrices with feature vectors for users and items.
    """
    model = AlternatingLeastSquares(factors=factors,
                                    regularization=regularization,
                                    iterations=iterations,
                                    use_native=False)
    model.fit((dataset * alpha_parameter).astype('double'))

    return model.item_factors, model.user_factors
