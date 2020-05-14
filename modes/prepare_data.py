import click
import numpy as np
import random
import pandas as pd
import scipy.sparse as sparse


@click.command(name='prepare-data', help='Transform raw dataset into format used for training.')
@click.option('--dataset', help='Path to ratings dataset.')
@click.option('--test_percentage', help='Percentage of user-item interactions that will be used in test set.')
@click.option('--seed', default=42, help='Random seed')
@click.option('--train_path', help='Path to training dataset location.')
@click.option('--test_path', help='Path to test dataset location.')
@click.option('--mask_path', help='Path to location of masked user indices.')
def prepare_data(**options):
    """Takes original data, creates user-item sparse matrix and masks percentage of the original ratings for test set.
    Test set will contain all of the original ratings, and train set will replace the specified percentage of them with
    a zero in original rating matrix.
    """
    dataset = pd.read_csv(options['dataset'])
    dataset = dataset.drop('timestamp')
    dataset['rating'] = dataset['rating'] * 2

    users = list(np.sort(dataset['userId'].unique()))
    rows = dataset['userId'].astype('category', categories=users).cat.codes
    movies = list(np.sort(dataset['movieId'].unique()))
    columns = dataset['movieId'].astype('category', categories=movies).cat.codes
    ratings = sparse.csr_matrix((dataset['rating'], (rows, columns)), shape=(len(users), len(movies)))

    # Create binary preference test matrix
    test_set = ratings.copy()
    test_set[test_set != 0] = 1

    # Create train set
    train_set = ratings.copy()
    # Get user-item interaction list
    nonzero_inds = train_set.nonzero()
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1]))

    random.seed(options['seed'])

    # Get random samples for test masking
    num_samples = int(np.ceil(options['test_percentage'] * len(nonzero_pairs)))
    samples = random.sample(nonzero_pairs, num_samples)

    # Mask test set in train matrix and remove zero user-item interactions from sparse matrix
    user_inds = [index[0] for index in samples]
    item_inds = [index[1] for index in samples]
    train_set[user_inds, item_inds] = 0
    train_set.eliminate_zeros()

    # Save train and tests datasets
    sparse.save_npz(options['train_path'], train_set)
    sparse.save_npz(options['test_path'], test_set)
    # Save masked user indices
    user_inds = pd.DataFrame(data=list(set(user_inds)), columns=['userInds'])
    user_inds.to_csv(options['mask_path'], index=False)
