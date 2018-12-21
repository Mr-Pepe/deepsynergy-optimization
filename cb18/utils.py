import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import pickle
from numpy import mean,cov,dot,linalg

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = {'x': self.X[idx], 'y': self.y[idx]}

        return sample


def show_solver_history(path):
    solver = pickle.load(open(path, 'rb'))

    train_loss = np.array(solver.train_loss_history)
    val_loss = np.array(solver.val_loss_history)

    f, (ax1, ax2) = plt.subplots(2,1)

    ax1.plot(train_loss)
    ax1.plot(np.convolve(train_loss, np.ones((1000,)) / 1000, mode='valid'))
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Train loss")

    ax2.plot(val_loss)
    ax2.plot(np.convolve(val_loss, np.ones((15,)) / 15, mode='valid'))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation loss")


    plt.show()


def normalize(X, means=None, std_devs=None, mask=None, tanh=False, reduce=False):
    if means is None:
        print("  Calculating means ... ", end='')
        means = X.mean(dim=0)
        print("Done.")

        print("  Calculating standard deviations ... ", end='')
        std_devs = X.std(dim=0)
        print("Done.")



    if reduce is False:
        mask = std_devs != 0
        X = X - means
        X[:, mask] = X[:, mask] / std_devs[mask]
    else:
        if mask is None:
            mask = std_devs != 0
            means = means[mask]
            std_devs = std_devs[mask]

        X = X[:,mask]
        X = X - means
        X = X / std_devs

    if tanh is True:
        X = X.tanh()

    return X, means, std_devs, mask


def mse(A,B):
    return ((A - B).pow(2).sum() / torch.numel(A)).item()

