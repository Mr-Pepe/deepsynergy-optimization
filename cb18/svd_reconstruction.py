import cb18.utils as utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

num_train_folds = 5
num_eigenvectors = 80

train_data_path     = ["../datasets/train_data_fold_%d.p" % i for i in range(5)]
svd_load_paths      = [["../datasets/svd_test_fold_%d_train_fold_%d.p" % (i_test_fold, i_train_fold) for i_train_fold in range(num_train_folds)] for i_test_fold in range(5)]


for i_test_fold in range(5):

    print("Loading train dataset for test fold %d... " % i_test_fold, end='')
    with open(train_data_path[i_test_fold], 'rb') as file:
        X, y = pickle.load(file)
    print("Done.")

    np.random.seed(0)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, num_train_folds)

    for i_train_fold in range(num_train_folds):

        train_indices = np.delete(indices, np.where(np.isin(indices, fold_indices[i_train_fold])))

        print("Loading V for test fold %d train fold %d ... " % (i_test_fold, i_train_fold), end='')
        with open(svd_load_paths[i_test_fold][i_train_fold], 'rb') as file:
            V = pickle.load(file)
        print("Done.")

        # Use tensor cloning to copy the data from the original tensor
        # It otherwise changes the original data during normalization
        X_train = X[train_indices].clone().detach()

        print("Normalize train data of test fold %d train fold %d ... " % (i_test_fold, i_train_fold))
        X_train, means, std_devs = utils.normalize(X_train, tanh=False)

        print("Reconstructing ...", end='')
        X_reconstructed = torch.matmul(X_train, V)

        print("MSE: " + str(utils.mse(X_train, X_reconstructed)))

