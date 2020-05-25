import os

import click
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler


@click.command(name='recommend-all', help='Recommend items to a user.')
@click.option('--dataset', help='Path to ratings dataset.')
@click.option('--train_path', help='Path to training set.')
@click.option('--vectors_path', help='Path to user and item vectors.')
@click.option('--recommendations', help='Path to output file.')
def recommend_all(**options):
    # Read sparse files
    train_set = sparse.load_npz(os.path.join(options['train_path'], 'train.npz'))
    user_vecs = sparse.load_npz(os.path.join(options['vectors_path'], 'user_vecs.npz'))
    item_vecs = sparse.load_npz(os.path.join(options['vectors_path'], 'item_vecs.npz'))

    # Get users and movies from original dataset
    dataset = pd.read_csv(options['dataset'])
    dataset = dataset.drop('timestamp', axis=1)
    dataset['rating'] = dataset['rating'] * 2
    user_list = np.sort(dataset['userId'].unique())
    item_list = np.sort(dataset['movieId'].unique())
    recommendations = pd.DataFrame(columns=['user_id', 'recommendations'])

    for user in user_list:
        # Select the index row of the user
        user_index = np.where(user_list == user)[0][0]
        # Get preference vector with ratings from the training set
        preference_vector = train_set[user_index, :].toarray()
        # Increment by one so we don't have zero items
        preference_vector = preference_vector.reshape(-1) + 1
        # Eliminate movies user already rated
        preference_vector[preference_vector > 1] = 0
        # Get recommendation vector from dot product of user vector and all item vectors
        recommendation_vector = user_vecs[user_index, :].dot(item_vecs)
        # Scale recommendations between 0 and 1
        recommendation_vector = MinMaxScaler().fit_transform(recommendation_vector.todense().reshape(-1, 1))[:, 0]
        recommendation_vector *= preference_vector
        # Sort item indices (already rated movies have their recommendation multiplied by zero)
        item_idx = np.argsort(recommendation_vector)[::-1]
        recommendation_list = [item_list[index] for index in item_idx]

        recommendations = recommendations.append({'user_id': user, 'recommendations': recommendation_list},
                                                 ignore_index=True)

    recommendations = recommendations.set_index(['user_id']).sort_index(axis=0)
    recommendations = np.array(recommendations['recommendations'].tolist())
    np.save(os.path.join(options['recommendations'], 'recommendations.npy'), recommendations)
