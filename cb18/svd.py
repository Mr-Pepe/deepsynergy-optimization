from matplotlib.mlab import PCA
import pickle
from cb18.utils import Dataset
import numpy as np
import torch
import cb18.utils as utils

train_data_path     = "../datasets/train_data.p"
fold_indices_path   = "../datasets/fold_indices.p"
svd_save_paths      = ["../datasets/svd%d.p" % i for i in range(4)]

print("Loading train dataset ... ", end='')
with open(train_data_path, 'rb') as file:
    X, y = pickle.load(file)
print("Done.")

print("Loading fold indices ...", end='')
with open(fold_indices_path, 'rb') as file:
    fold_indices = pickle.load(file)
print("Done.")

# Calculate the SVD on the train data sets according to the folding indices
for i in range(1,4):
    train_indices = fold_indices[i][0]

    # Use tensor cloning to copy the data from the original tensor
    # It otherwise changes the original data during normalization
    X_train = X[train_indices].clone().detach()

    print("Normalize train data of fold %d ... " % i)
    X_train, means, std_devs = utils.normalize(X_train, tanh=False)

    print("Doing SVD decomposition for fold %d ... " % i, end='')
    U,S,V = torch.svd(X_train)
    print("Done.")

    del X_train

    print("Saving U,S,V for fold %d ... " % i, end='')
    with open(svd_save_paths[i], 'wb') as file:
        pickle.dump((U,S,V), file)
    print("Done.")