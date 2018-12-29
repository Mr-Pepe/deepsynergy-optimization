from matplotlib.mlab import PCA
import pickle
from cb18.utils import Dataset
import numpy as np
import torch
import cb18.utils as utils
import matplotlib.pyplot as plt

config = {
    'train_data_path':  '../datasets/train_data.p',

}

print("Loading train dataset ... ", end='')
with open(config['train_data_path'], 'rb') as train_data_file:
    X, y = pickle.load(train_data_file)
    print("Done.")

X, means, std_devs = utils.normalize(X, tanh=False)

idx = std_devs.argsort()
plt.plot(std_devs[idx])



print("Done")