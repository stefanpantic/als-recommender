import click
from scipy import sparse


@click.command(name='train', help='Train ALS Recommender.')
@click.option('--algorithm', required=True, type=click.Choice(['lib', 'impl']), help='Whether to use implicit library or algorithm implementation.')
@click.option('--train_set', required=True, help='Path to training dataset.')
@click.option('--regularization', default=0.1, help='Regularization parameter (learning rate).')
@click.option('--iterations', default=50, help='Number of iterations.')
@click.option('--factors', default=20, help='Number of user and item factors.')
@click.option('--alpha', default=40, help='Alpha parameter (for creating confidence matrix).')
@click.option('--seed', default=42, help='Random seed.')
@click.option('--user_path', default='./logs/als/user_vecs.npz', help='Where to log result user vecs.')
@click.option('--item_path', default='./logs/als/item_vecs.npz', help='Where to log result item vecs.')
def train(**options):
    if options['algorithm'] == 'impl':
        from models.als_implementation import als_algorithm as als
    elif options['algorithm'] == 'lib':
        from models.als_implicit import get_implicit_als as als
    else:
        raise NotImplementedError

    dataset = sparse.load_npz(options['train_set'])
    als_params = {
        'dataset': dataset,
        'regularization': options['regularization'],
        'alpha_parameter': options['alpha'],
        'iterations': options['iterations'],
        'factors': options['factors'],
        'random_state': options['seed'],
    }

    user_vecs, item_vecs = als(**als_params)
    sparse.save_npz(options['user_path'], user_vecs)
    sparse.save_npz(options['item_path'], item_vecs)

