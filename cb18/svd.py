from matplotlib.mlab import PCA
import pickle
from cb18.utils import Dataset
import numpy as np
import torch
import cb18.utils as utils

config = {
    'train_data_path':  '../datasets/train_data.p',

}

print("Loading train dataset ... ", end='')
with open(config['train_data_path'], 'rb') as train_data_file:
    X, y = pickle.load(train_data_file)
    print("Done.")

print("Splitting dataset ... ", end='')
num_train = int(len(X)*0.8)
num_val = len(X)-num_train
torch.manual_seed(123)
[train_set, val_set] = torch.utils.data.random_split(X, (num_train, num_val))
print("Done.")

print("Doing SVD decomposition ... ", end='')
X_train = X[train_set.indices]
X_train, means, std_devs, _ = utils.normalize(X_train, tanh=False)

U,S,V = torch.svd(X_train)
print("Done.")

print("Saving U,S,V ... ", end='')
with open("../datasets/svd.p", 'wb') as file:
    pickle.dump((U,S,V), file)
print("Done.")