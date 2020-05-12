import click


@click.command(name='train', help='Train ALS Recommender.')
@click.option('--algorithm', required=True, type=click.Choice(['lib', 'impl']), help='Whether to use implicit library or algorithm implementation.')
@click.option('--dataset', required=True, help='Path to training dataset.')
@click.option('--regularization', default=0.1, help='Regularization parameter (learning rate).')
@click.option('--iterations', default=50, help='Number of iterations.')
@click.option('--factors', default=20, help='Number of user and item factors.')
@click.option('--alpha', default=40, help='Alpha parameter (for creating confidence matrix).')
@click.option('--seed', default=42, help='Random seed.')
@click.option('--log_dir', default='./logs/als', help='Where to log weights.')
def train(**options):
    pass
