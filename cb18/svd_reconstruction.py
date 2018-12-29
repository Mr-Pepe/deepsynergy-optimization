import cb18.utils as utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

train_data_path   = "../datasets/train_data.p"
fold_indices_path = "../datasets/fold_indices.p"
svd_save_paths    = ["../datasets/svd%d.p" % i for i in range(4)]

print("Loading train dataset ... ", end='')
with open(train_data_path, 'rb') as train_data_file:
    X, y = pickle.load(train_data_file)
print("Done.")

print("Loading fold indices ...", end='')
with open(fold_indices_path, 'rb') as file:
    fold_indices = pickle.load(file)
print("Done.")


for i in range(4):

    train_indices = fold_indices[i][0]

    print("Loading U, S, V for fold %d ... " % i, end='')
    with open(svd_save_paths[i], 'rb') as file:
        U, S, V = pickle.load(file)
    print("Done.")

    # Use tensor cloning to copy the data from the original tensor
    # It otherwise changes the original data during normalization
    X_train = X[train_indices].clone().detach()

    print("Normalize train data of fold %d ... " % i)
    X_train, means, std_devs = utils.normalize(X_train, tanh=False)

    var = torch.cumsum(S.pow(2) / S.pow(2).sum(), 0)

    for k in [1, 10, 50, 80, 106, 107, 108, 200, 500]:
        print("Reconstructing with k=" + str(k), " ... ", end='')
        X_reconstructed = torch.matmul(torch.matmul(U[:, :k], torch.diag(S[:k])), torch.t(V[:, :k]))

        print("MSE: " + str(utils.mse(X_train, X_reconstructed)))

    plt.plot(var[:200].numpy())
    plt.xlabel("Ordered Eigenvectors")
    plt.ylabel("Covered Variance")
    plt.show()
