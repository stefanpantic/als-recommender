import click
import pandas as pd
from scipy import sparse
import numpy as np
from utilities.auc import calculate_mean_auc
import os


@click.command(name='summary', help='Get recommender summary (AUC Score).')
@click.option('--dataset_path', help='Path to datasets.')
@click.option('--recommendations', help='Path to recommendations.')
def summary(**options):
    train_set = sparse.load_npz(os.path.join(options['dataset_path'], 'train.npz'))
    test_set = sparse.load_npz(os.path.join(options['dataset_path'], 'test.npz'))
    masked_users = list(pd.read_csv(os.path.join(options['dataset_path'], 'mask.csv'))['userInds'])
    recommendations = np.load(os.path.join(options['recommendations'], 'recommendations.npy'))

    mean_auc, mean_popular_auc = calculate_mean_auc(train_set=train_set,
                                                    test_set=test_set,
                                                    test_users=masked_users,
                                                    predictions=recommendations)

    print(f'Mean AUC value: {mean_auc}')
    print(f'Mean popular AUC value: {mean_popular_auc}')
