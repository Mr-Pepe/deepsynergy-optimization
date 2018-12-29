import cb18.utils as utils
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

# Activate interactive mode
plt.ion()

config = {
    'train_data_path':  '../datasets/train_data.p',
}

print("Loading U,S,V ...", end='')
with open("../datasets/svd.p", "rb") as file:
    U,S,V = pickle.load(file)
print("Done.")

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

X_train = X[train_set.indices]
X_train, means, std_devs = utils.normalize(X_train, tanh=False)

var = torch.cumsum(S.pow(2)/S.pow(2).sum(), 0)


for k in [1,10,50,80,106, 200, 500]:
    print("Reconstructing with k=" + str(k), " ... ", end='')
    X_reconstructed = torch.matmul(torch.matmul(U[:,:k],torch.diag(S[:k])),torch.t(V[:,:k]))

    print("MSE: " + str(utils.mse(X_train, X_reconstructed)))


plt.plot(var[:200].numpy())
plt.xlabel("Ordered Eigenvectors")
plt.ylabel("Covered Variance")
plt.draw()
plt.pause(0.05)

