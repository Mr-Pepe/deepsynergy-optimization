from matplotlib.mlab import PCA
import pickle
from cb18.utils import Dataset
import numpy as np
import torch
import cb18.utils as utils


num_train_folds = 5
num_eigenvectors = 80

train_data_path     = ["../datasets/train_data_fold_%d.p" % i for i in range(5)]
svd_save_paths      = [["../datasets/svd_test_fold_%d_train_fold_%d.p" % (i_test_fold, i_train_fold) for i_train_fold in range(num_train_folds)] for i_test_fold in range(5)]

for i_test_fold in range(5):

    print("Loading train dataset for test fold %d... " % i_test_fold, end='')
    with open(train_data_path[i_test_fold], 'rb') as file:
        X, y = pickle.load(file)
    print("Done.")

    np.random.seed(0)
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, num_train_folds)

    # Calculate the SVD on the train data sets according to the folding indices
    for i_train_fold in range(num_train_folds):

        train_indices = np.delete(indices, np.where(np.isin(indices, fold_indices[i_train_fold])))



        # Use tensor cloning to copy the data from the original tensor
        # It otherwise changes the original data during normalization
        X_train = X[train_indices].clone().detach()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Using device: " + device.type)

        X_train.to(device)

        print("Normalize train data of test fold %d train fold %d ... " % (i_test_fold, i_train_fold))
        X_train, means, std_devs = utils.normalize(X_train, tanh=False)

        print("Doing SVD decomposition for test fold %d train fold %d ... " % (i_test_fold, i_train_fold), end='')
        U, S, V = X_train.svd()
        print("Done.")

        del X_train

        # Only save the num_eigenvectors most important vectors of V for memory saving
        print("Saving V for test fold %d train fold %d ... " % (i_test_fold, i_train_fold), end='')
        with open(svd_save_paths[i_test_fold][i_train_fold], 'wb') as file:
            pickle.dump(V[:, :num_eigenvectors], file)
        print("Done.")
