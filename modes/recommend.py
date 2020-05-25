import os

import click
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

from utilities.get_user_items import get_user_items


@click.command(name='recommend', help='Recommend items to a user.')
@click.option('--user_id', required=True, type=int, help='Id of the user.')
@click.option('--dataset', help='Path to ratings dataset.')
@click.option('--train_path', help='Path to training set.')
@click.option('--vectors_path', help='Path to user and item vectors.')
@click.option('--num_items', default=1, help='Number of recommended items.')
@click.option('--movie_dataset', help='Path to movie dataset for getting movie names.')
def recommend(**options):
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
    most_liked = dataset.loc[dataset['userId'] == options['user_id']].sort_values(by='rating', ascending=False)[
                     'movieId'][:3]

    # Select the index row of the user
    user_index = np.where(user_list == options['user_id'])[0][0]
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
    item_idx = np.argsort(recommendation_vector)[::-1][:options['num_items']]
    recommendation_list = [item_list[index] for index in item_idx]

    # Get movie titles for recommended movies
    movies = pd.read_csv(options['movie_dataset'])
    recommendation_list = movies.loc[movies['movieId'].isin(recommendation_list)]['title'].tolist()
    most_liked = movies.loc[movies['movieId'].isin(most_liked)]['title'].tolist()
    watched_movies = get_user_items(user_id=options['user_id'], train_matrix=train_set,
                                    user_list=user_list, item_list=item_list)

    watched_ratings = dataset.loc[dataset['movieId'].isin(watched_movies)]['rating'].tolist()
    watched_movies = movies.loc[movies['movieId'].isin(watched_movies)]['title'].tolist()

    print(f"User with id {options['user_id']} previously watched these movies and gave them following ratings:")
    for movie, rating in list(zip(watched_movies, watched_ratings)):
        print(f'- {movie}: {rating}')

    print()
    print('They liked:')
    for num, movie in enumerate(most_liked):
        print(f'{num + 1}. {movie}')

    print('the most, so we would recommend them these movies:')
    for num, movie in enumerate(recommendation_list):
        print(f'{num + 1}. {movie}')

    np.save(os.path.join(options['vectors_path'], 'recommendations.npy'), recommendation_list)
