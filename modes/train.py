import os

import click
from scipy import sparse


@click.command(name='train', help='Train ALS Recommender.')
@click.option('--algorithm', required=True, type=click.Choice(['lib', 'impl']),
              help='Whether to use implicit library or algorithm implementation.')
@click.option('--dataset_path', required=True, help='Path to training dataset.')
@click.option('--regularization', default=0.1, help='Regularization parameter (learning rate).')
@click.option('--iterations', default=50, help='Number of iterations.')
@click.option('--factors', default=20, help='Number of user and item factors.')
@click.option('--alpha', default=40, help='Alpha parameter (for creating confidence matrix).')
@click.option('--seed', default=42, help='Random seed.')
@click.option('--output_path', default='./logs/als/', help='Where to log result user vecs.')
def train(**options):
    if options['algorithm'] == 'impl':
        from models.als_implementation import als_algorithm as als
    elif options['algorithm'] == 'lib':
        from models.als_implicit import get_implicit_als as als
    else:
        raise NotImplementedError

    dataset = sparse.load_npz(os.path.join(options['dataset_path'], 'train.npz'))
    als_params = {
        'dataset': dataset,
        'regularization': options['regularization'],
        'alpha_parameter': options['alpha'],
        'iterations': options['iterations'],
        'factors': options['factors'],
        'random_state': options['seed'],
    }

    user_vecs, item_vecs = als(**als_params)
    if not os.path.exists(options['output_path']):
        os.makedirs(options['output_path'], exist_ok=True)

    sparse.save_npz(os.path.join(options['output_path'], 'user_vecs.npz'), user_vecs)
    sparse.save_npz(os.path.join(options['output_path'], 'item_vecs.npz'), item_vecs)
