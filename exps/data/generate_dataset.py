import argparse
import socket

import torch
import numpy as np
from sklearn.datasets import make_sparse_spd_matrix
import random


def generate_dataset_realistic(
        n=100,
        p=10,
        density=0.7,
        dataset_size=1,
        as_tensor=True,
        random_state=0,
        same_Theta_true=False):

    rng = np.random.RandomState(random_state)

    dataset = []
    for i in range(dataset_size):

        alpha = density

        Theta_true = make_sparse_spd_matrix(
            p,
            alpha=alpha,
            random_state=random_state if same_Theta_true else rng
        )
        # Improve Theta_true conditioning
        Theta_true.flat[::p+1] += 1e-1

        # Real covariance matrix
        Sigma_true = np.linalg.pinv(Theta_true)

        # iid centered samples
        X = rng.multivariate_normal(
            mean=np.zeros(p),
            cov=Sigma_true,
            size=n
        )

        # Empirical covariance matrix
        S = np.cov(X, bias=True, rowvar=False)

        # Convert arrays to tensors
        if as_tensor:
            S = torch.tensor(S)
            Theta_true = torch.tensor(Theta_true)
            Sigma_true = torch.tensor(Sigma_true)

        dataset.append((S, Theta_true, Sigma_true, X, alpha))

    return dataset


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate datasets')

    parser.add_argument('-set_type',
                        '--set_type',
                        nargs='?',
                        required=True,
                        type=str,
                        help='Train/test/validation set.')

    parser.add_argument('-n',
                        '--number_of_data_samples',
                        nargs='?',
                        default="",
                        type=int,
                        help='The number of samples to use to compute the empirical covariance matrix.',
                        required=True)

    parser.add_argument('-p',
                        '--dimension_of_data_samples',
                        nargs='?',
                        default="",
                        type=int,
                        help='The dimension of samples to use to compute the empirical covariance matrix.',
                        required=True)

    parser.add_argument('-den',
                        '--density',
                        nargs='?',
                        default=0.7,
                        type=float,
                        help='Controls the sparsity degree of true precision matrix.',
                        required=True)

    parser.add_argument('-size',
                        '--dataset_size',
                        nargs='?',
                        default="",
                        type=int,
                        help='Size of the dataset.',
                        required=True)

    parser.add_argument('-as_tensor',
                        '--as_tensor',
                        nargs='?',
                        default=1,
                        choices=[0, 1],
                        type=int,
                        required=True,
                        help='Controls whether to use arrays or tensors as datatype.')

    parser.add_argument('-random_state',
                        '--random_state',
                        nargs='?',
                        default=0,
                        type=int,
                        help='Seed for data generation.',
                        required=True)

    parser.add_argument('-same_Theta_true',
                        '--same_Theta_true',
                        nargs='?',
                        default=0,
                        choices=[0, 1],
                        type=int,
                        required=True,
                        help='Controls whether the empirical covariance matrices are generated using the same precision matrix.')

    print(socket.gethostname())

    args = parser.parse_args()

    dataset = generate_dataset_realistic(n=args.number_of_data_samples,
                                         p=args.dimension_of_data_samples,
                                         density=args.density,
                                         dataset_size=args.dataset_size,
                                         as_tensor=args.as_tensor,
                                         random_state=args.random_state,
                                         same_Theta_true=args.same_Theta_true)

    path = f'./{args.set_type}/p_{args.dimension_of_data_samples}_n_{args.number_of_data_samples}_density_{args.density}_size_{args.dataset_size}_random_state_{args.random_state}.pt'
    torch.save(dataset, path)
    print("Saved dataset.")
