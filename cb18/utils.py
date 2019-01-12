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
    history, stop_reason, stop_time = pickle.load(open(path, 'rb'))

    print("Stop reason: %s" %stop_reason)
    print("Stop time: %fs" %stop_time)

    train_loss = np.array(history['train_loss_history'])
    val_loss = np.array(history['val_loss_history'])

    f, (ax1, ax2, ax3) = plt.subplots(3,1)

    ax1.plot(train_loss)
    ax1.plot(np.convolve(train_loss, np.ones((460,)) / 460, mode='valid'))
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Train loss")

    ax2.plot(val_loss)
    ax2.plot(np.convolve(val_loss, np.ones((15,)) / 15, mode='valid'))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation loss")

    ax3.plot(np.arange(0,len(val_loss))*460,val_loss)
    ax3.plot(np.convolve(train_loss, np.ones((460,)) / 460, mode='valid'))
    # ax3.plot(np.arange(0,len(val_loss))*460,np.convolve(val_loss, np.ones((15,)) / 15, mode='valid'))

    plt.show()


def normalize(X, means=None, std_devs=None, tanh=False):
    if means is None:
        print("  Calculating means ... ", end='')
        means = X.mean(dim=0)
        print("Done.")

        print("  Calculating standard deviations ... ", end='')
        std_devs = X.std(dim=0)
        print("Done.")

    mask = std_devs != 0
    X = X - means
    X[:, mask] = X[:, mask] / std_devs[mask]

    if tanh is True:
        X = X.tanh()

    return X, means, std_devs


def mse(A,B):
    return ((A - B).pow(2).sum() / torch.numel(A)).item()

