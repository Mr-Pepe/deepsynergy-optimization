import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import pickle


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

    loss = np.array(solver.train_loss_history)

    plt.plot(loss[60000:])
    plt.plot(np.convolve(loss[60000:], np.ones((1000,)) / 1000, mode='valid'))
    plt.xlabel("Iterations")
    plt.ylabel("Train loss")
    plt.show()


def normalize(X, means=None, std_devs=None):
    if means is None:
        means = X.mean(dim=0)
        std_devs = X.std(dim=0)

    mask = std_devs != 0

    X = X - means
    X[:, mask] = X[:, mask] / std_devs[mask]
    X = X.tanh()

    return X, means, std_devs