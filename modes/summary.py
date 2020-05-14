import click
import pandas as pd
from scipy import sparse

from utilities.auc import calculate_mean_auc


@click.command(name='summary', help='Get recommender summary (AUC Score).')
@click.option('--train_set', help='Path to train set.')
@click.option('--test_set', help='Path to test set.')
@click.option('--recommendations', help='Path to recommendations.')
@click.option('--masked_users', help='Path to list of users masked for test set.')
def summary(**options):
    train_set = sparse.load_npz(options['train_set'])
    test_set = sparse.load_npz(options['test_set'])

    # TODO: Read recommendations
    recommendations = None
    masked_users = pd.read_csv(options['masked_users'])['userInds'].toarray()

    mean_auc, mean_popular_auc = calculate_mean_auc(train_set=train_set,
                                                    test_set=test_set,
                                                    test_users=masked_users,
                                                    predictions=recommendations)

    print(f'Mean AUC value: {mean_auc}')
    print(f'Mean popular AUC value: {mean_popular_auc}')

