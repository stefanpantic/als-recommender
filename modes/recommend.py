import click
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from scipy import sparse


@click.command(name='recommend', help='Recommend items to a user.')
@click.option('--user_id', required=True, help='Id of the user.')
@click.option('--train_set', help='Path to training set.')
@click.option('--user_vecs', help='Path to user vectors.')
@click.option('--item_vecs', help='Path to item vectors.')
@click.option('--num_items', default=1, help='Number of recommended items.')
@click.option('--movie_dataset', help='Path to movie dataset for getting movie names.')
def recommend(**options):
    train_set = sparse.load_npz(options['train_set'])
    user_vecs = sparse.load_npz(options['user_vecs'])
    item_vecs = sparse.load_npz(options['item_vecs'])

    # TODO: Get user and item lists
    user_list = []
    item_list = []

    # Select the index row of the user
    user_index = np.where(user_list == options['user_id'])[0][0]
    # Get preference vector with ratings from the training set
    preference_vector = train_set[user_index, :].toarray()
    # Increment by one so we don't have zero items
    preference_vector = preference_vector.reshape(-1) + 1
    # Eliminate movies user already rated
    preference_vector[preference_vector > 1] = 0
    # Get recommendation vector from dot product of user vector and all item vectors
    recommendation_vector = user_vecs[user_index, :].dot(item_vecs.T)
    # Scale recommendations between 0 and 1
    recommendation_vector = MinMaxScaler().fit_transform(recommendation_vector.reshape(-1, 1))[:, 0]
    recommendation_vector *= preference_vector
    # Sort item indices (already rated movies have their recommendation multiplied by zero)
    item_idx = np.argsort(recommendation_vector)[::-1][:options['num_items']]
    recommendation_list = [item_list[index] for index in item_idx]

    # Get movie titles for recommended movies
    movies = pd.read_csv(options['movie_dataset'])
    recommendation_list = movies[movies['movieId' in recommendation_list]]['title'].toarray()

    print(recommendation_list)

    return recommendation_list
