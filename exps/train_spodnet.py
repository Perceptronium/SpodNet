import argparse
import socket
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

from spodnet import SpodNet


def save_data(avg_train,
              avg_test,
              std_train,
              std_test,
              avg_train_f1_scores,
              std_train_f1_scores,
              avg_test_f1_scores,
              std_test_f1_scores,
              path,
              net):
    print("Saving data...")
    torch.save(avg_train, path + f'avg_train_losses.pt')
    torch.save(avg_test, path + f'avg_test_losses.pt')
    torch.save(std_train, path + f'std_train_losses.pt')
    torch.save(std_test, path + f'std_test_losses.pt')
    torch.save(avg_train_f1_scores, path + f'avg_train_f1_scores.pt')
    torch.save(std_train_f1_scores, path + f'std_train_f1_scores.pt')
    torch.save(avg_test_f1_scores, path + f'avg_test_f1_scores.pt')
    torch.save(std_test_f1_scores, path + f'std_test_f1_scores.pt')

    print("Data is saved.")
    print("Saving trained model...")
    torch.save(net, path + f'trained_model.pt')
    print("Model is saved.")
    return avg_train, avg_test, std_train, std_test, path


def train_spodnet(train_samples,
                  test_samples,
                  n,
                  p,
                  train_batch_size,
                  test_batch_size,
                  precision_sparsity,
                  K,
                  training_seed,
                  epochs,
                  lr,
                  learning_mode):

    # Datasets
    train_matrices = torch.load(
        f'./data/train/p_{p}_n_{n}_density_{precision_sparsity}_size_{10_000}_random_state_{0}.pt')
    train_set = train_matrices[:train_samples]

    test_matrices = torch.load(
        f'./data/test/p_{p}_n_{n}_density_{precision_sparsity}_size_{10_000}_random_state_{1}.pt')
    test_set = test_matrices[:test_samples]

    # Dataloader
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=train_batch_size,
        shuffle=1,
        num_workers=4
    )

    test_loader = DataLoader(
        dataset=test_set,
        batch_size=test_batch_size,
        shuffle=0,
        num_workers=4
    )

    device = "cpu"
    print(f"Training on {device}.")

    torch.manual_seed(training_seed)
    net = SpodNet(K=K,
                  p=p,
                  layer_type=f'{learning_mode}',
                  device=device)

    net.to(device)

    nb_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Network has {nb_params} learnable parameters.")

    optimizer = Adam(net.parameters(), lr=lr)

    avg_train_losses = []
    std_train_losses = []
    avg_train_f1_scores_list = []
    std_train_f1_scores_list = []
    avg_test_f1_scores_list = []
    std_test_f1_scores_list = []
    avg_test_losses = []
    std_test_losses = []

    # Saving directory
    path = f"./{learning_mode}/"
    Path(path).mkdir(parents=True, exist_ok=True)

    # Training loop
    print(
        f"========== Training masked network: batch_size={train_batch_size}, K={K}. ========== ")

    for epoch in range(epochs):

        print(f"K={K} [{epoch}/{epochs}]")

        net.train()

        train_individual_NMSE_losses_list = torch.tensor([])
        train_individual_f1_losses = torch.tensor([])

        for i, (S, Theta_true, _, _, _) in enumerate(train_loader):
            Theta_true = Theta_true.to(device)

            pred = net(S)

            train_individual_batch_losses = (
                torch.linalg.matrix_norm(pred - Theta_true, ord='fro')**2)

            torch.mean(train_individual_batch_losses).backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging the NMSEs
            train_individual_NMSE_losses = (
                train_individual_batch_losses /
                torch.linalg.matrix_norm(Theta_true, ord='fro')**2)
            train_individual_NMSE_losses_list = torch.cat(
                (train_individual_NMSE_losses_list, train_individual_NMSE_losses.detach()))

            # Logging the F1 scores
            for (T_t, T) in zip(Theta_true, pred):
                T_t_support = T_t.detach().numpy().copy()
                T_t_support[T_t != 0.] = 1.
                T_support = torch.round(
                    T, decimals=10).detach().numpy().copy()
                T_support[T_support != 0.] = 1.
                individual_f1 = f1_score(T_t_support.flatten(),
                                         T_support.flatten())
                train_individual_f1_losses = torch.cat(
                    (train_individual_f1_losses, torch.tensor([individual_f1])))

        avg_train_losses.append(torch.mean(
            train_individual_NMSE_losses_list).item())
        std_train_losses.append(
            train_individual_NMSE_losses_list.std().item())
        avg_train_f1_scores_list.append(train_individual_f1_losses.mean())
        std_train_f1_scores_list.append(train_individual_f1_losses.std())

        # Evaluate model on test data
        net.eval()
        test_individual_NMSEs = torch.tensor([])
        test_individual_f1_losses = torch.tensor([])

        with torch.no_grad():
            for (S, Theta_true, _, _, _) in test_loader:
                S = S.to(device)

                Theta_true = Theta_true.to(device)

                pred = net(S)

            # Logging the NMSEs
            test_individual_NMSE_losses = (
                torch.linalg.matrix_norm(pred - Theta_true, ord='fro')**2 /
                torch.linalg.matrix_norm(Theta_true, ord='fro')**2)
            test_individual_NMSEs = torch.cat(
                (test_individual_NMSEs, test_individual_NMSE_losses.detach()))

            # Logging the F1 scores
            for (T_t, T) in zip(Theta_true, pred):
                T_t_support = T_t.detach().numpy().copy()
                T_t_support[T_t != 0.] = 1.
                T_support = torch.round(
                    T, decimals=10).detach().numpy().copy()
                T_support[T_support != 0.] = 1.
                individual_f1 = f1_score(T_t_support.flatten(),
                                         T_support.flatten())
                test_individual_f1_losses = torch.cat(
                    (test_individual_f1_losses, torch.tensor([individual_f1])))

        avg_test_losses.append(torch.mean(
            test_individual_NMSEs).item())
        std_test_losses.append(
            test_individual_NMSEs.std().item())
        avg_test_f1_scores_list.append(test_individual_f1_losses.mean())
        std_test_f1_scores_list.append(test_individual_f1_losses.std())

        print(
            f"Avg NMSE train loss : {torch.mean(train_individual_NMSE_losses_list)}")
        print(
            f"Avg F1 train score : {torch.mean(train_individual_f1_losses)}")
        print(
            f"Avg NMSE test loss : {torch.mean(test_individual_NMSEs)}")
        print(
            f"Avg F1 test score : {torch.mean(test_individual_f1_losses)}")

    print("========== Training masked network complete. ========== ")

    return save_data(avg_train_losses,
                     avg_test_losses,
                     std_train_losses,
                     std_test_losses,
                     avg_train_f1_scores_list,
                     std_train_f1_scores_list,
                     avg_test_f1_scores_list,
                     std_test_f1_scores_list,
                     path,
                     net)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train SpodNet')
    parser.add_argument('-train_samples',
                        '--train_samples',
                        nargs='?',
                        required=True,
                        type=int,
                        help='Number of samples to train on.')
    parser.add_argument('-test_samples',
                        '--test_samples',
                        nargs='?',
                        default="",
                        type=int,
                        help='Number of samples to test on.',
                        required=True)
    parser.add_argument('-p',
                        '--dimension_of_data_samples',
                        nargs='?',
                        default="",
                        type=int,
                        help='The dimension of samples to use to compute the empirical covariance matrix.',
                        required=True)
    parser.add_argument('-n',
                        '--number_of_data_samples',
                        nargs='?',
                        default="",
                        type=int,
                        help='The number of samples to use to compute the empirical covariance matrix.',
                        required=True)
    parser.add_argument('-train_batch_size',
                        '--train_batch_size',
                        nargs='?',
                        type=int,
                        help='Batch size for training dataloader.',
                        required=True)
    parser.add_argument('-test_batch_size',
                        '--test_batch_size',
                        nargs='?',
                        type=int,
                        help='Batch size for testing dataloader.',
                        required=True)
    parser.add_argument('-precision_sparsity',
                        '--precision_sparsity',
                        nargs='?',
                        type=float,
                        help='Sparsity degree of true precision matrices.',
                        required=True)
    parser.add_argument('-K',
                        '--K',
                        nargs='?',
                        type=int,
                        help='Number of unrolled iterations.',
                        required=True)
    parser.add_argument('-training_seed',
                        '--training_seed',
                        nargs='?',
                        type=int,
                        default=101,
                        help='Trainign seed.',
                        )
    parser.add_argument('-epochs',
                        '--epochs',
                        nargs='?',
                        type=int,
                        help='Number of Adam epochs to train for.',
                        required=True)
    parser.add_argument('-lr',
                        '--lr',
                        nargs='?',
                        type=float,
                        help='Learning rate of Adam.',
                        required=True)
    parser.add_argument('-learning_mode',
                        '--learning_mode',
                        nargs='?',
                        help='UBG, PNP or E2E.',
                        required=True)

    print(socket.gethostname())

    args = parser.parse_args()

    (avg_train_losses,
     avg_test_losses,
     std_train_losses,
     std_test_losses,
     path) = train_spodnet(args.train_samples,
                           args.test_samples,
                           args.number_of_data_samples,  # n
                           args.dimension_of_data_samples,  # p
                           args.train_batch_size,
                           args.test_batch_size,
                           args.precision_sparsity,
                           args.K,
                           args.training_seed,
                           args.epochs,
                           args.lr,
                           args.learning_mode)

    plot_results = False
    if plot_results:
        plt.close('all')
        fig, ax = plt.subplots(1, 1)
        ax.semilogy(avg_test_losses, linestyle='solid', color="tab:blue", lw=2,
                    label=f"{args.learning_mode.upper()} test")
        ax.fill_between(x=np.arange(len(avg_test_losses)),
                        y1=np.array(avg_test_losses) +
                        np.array(std_test_losses),
                        y2=np.array(avg_test_losses)-np.array(std_test_losses), color="tab:blue", alpha=0.2)
        ax.set_xlabel("Epochs")
        ax.set_xlabel("NMSE")
        ax.grid(which='both', alpha=0.9)
        ax.legend()
        plt.show()
