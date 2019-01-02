from matplotlib.mlab import PCA
import pickle
from cb18.utils import Dataset
import numpy as np
import torch
import cb18.utils as utils

train_data_path     = ["../datasets/train_data_fold_%d.p" % i for i in range(5)]
svd_save_paths      = ["../datasets/svd_fold_%d.p" % i for i in range(5)]

for i_test_fold in range(5):

    print("Loading train dataset for test fold %d... " % i_test_fold, end='')
    with open(train_data_path[i_test_fold], 'rb') as file:
        X, y = pickle.load(file)
    print("Done.")


    # Calculate the SVD on the train data sets according to the folding indices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device: " + device.type)

    X.to(device)

    print("Normalize train data of test fold %d ... " % i_test_fold)
    X, means, std_devs = utils.normalize(X, tanh=False)

    print("Doing SVD decomposition of train data for test fold  ... " % i_test_fold, end='')
    U, S, V = X.svd()
    print("Done.")

    del X

    print("Saving V for test fold %d ... " % i_test_fold, end='')
    with open(svd_save_paths[i_test_fold], 'wb') as file:
        pickle.dump(V, file)
    print("Done.")
