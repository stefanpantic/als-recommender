import click


@click.command(name='summary', help='Get recommender summary (AUC Score).')
@click.option('--train_set', help='Path to train set.')
@click.option('--test_set', help='Path to test set.')
@click.option('--recommendations', help='Path to recommendations.')
@click.option('--masked_users', help='Path to list of users masked for test set.')
def summary(**options):
    pass
