import torch
from cb18.model import SynergyNetwork
from cb18.solver import Solver
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import pickle
from cb18.utils import Dataset
import cb18.utils as utils
import datetime
import os
import yaml
import random

config = {
    'train_data_path':  '../datasets/train_data.p',
    'test_data_path': '../datasets/test_data.p',

    'num_eigenvectors': 80,
    'norm_tanh': False,
    'patience': 50,

    ## Hyperparameters ##
    'num_epochs':   5000,                # Number of epochs to train
    'batch_size':   [1, 64, 256, 1024],
    'n_hidden_1':   [8182, 2048, 1024, 512],
    'n_hidden_2':   [4096, 1024, 512],
    'learning_rate': [1e-5, 1e-4, 1e-3],
    'batch_norm':   [False],
    'dropout':      [0.5],
    'betas': (0.9, 0.999),             # Beta coefficients for ADAM
    'lr_decay': [1],                     # Learning rate decay -> lr *= lr_decay
    'lr_decay_interval': [1500],         # Number of epochs after which to reduce the learning rate

    ## Logging ##
    'log_interval': 1e15,           # Number of mini-batches after which to print training loss
    'save_interval': 10000,         # Number of epochs after which to save model and solver
    'save_path': '../saves'
}


print("Loading train dataset ... ", end='')
with open(config['train_data_path'], 'rb') as train_data_file:
    X, y = pickle.load(train_data_file)
    dataset = Dataset(X, y)
print("Done.")

print("Splitting dataset ... ", end='')
num_train = int(len(dataset)*0.8)
num_val = len(dataset)-num_train
torch.manual_seed(123)
[train_set, val_set] = torch.utils.data.random_split(dataset, (num_train, num_val))

X_train = X[train_set.indices]
y_train = y[train_set.indices]
X_val   = X[val_set.indices]
y_val   = y[val_set.indices]
print("Done.")

print("Normalizing dataset ... ")
X_train, means, std_devs = utils.normalize(X_train, tanh=config['norm_tanh'])
print("Done.")

print("Loading V matrix ...", end='')
with open("../datasets/svd.p", "rb") as file:
    _,_,V = pickle.load(file)
    V = V[:,:config['num_eigenvectors']]
print("Done.")

print("Projecting train data ... ", end='')
X_train = torch.matmul(X_train, V)
print("Done. ")

train_set = Dataset(X_train, y_train)
val_set   = Dataset(X_val, y_val)

train_data_sampler  = SubsetRandomSampler(range(len(train_set)))
val_data_sampler    = SubsetRandomSampler(range(len(val_set)))

random.shuffle(config['n_hidden_1'])
random.shuffle(config['n_hidden_2'])
random.shuffle(config['batch_norm'])
random.shuffle(config['dropout'])
random.shuffle(config['batch_size'])
random.shuffle(config['learning_rate'])
random.shuffle(config['lr_decay'])
random.shuffle(config['lr_decay_interval'])

for n_hidden_1 in config['n_hidden_1']:
    for n_hidden_2 in config['n_hidden_2']:
        for batch_norm in config['batch_norm']:
            for dropout in config['dropout']:
                for batch_size in config['batch_size']:
                    for lr in config['learning_rate']:
                        for lr_decay in config['lr_decay']:
                            for lr_decay_interval in config['lr_decay_interval']:

                                train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                                                batch_size=batch_size,
                                                                                sampler=train_data_sampler)
                                val_data_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size,
                                                                              sampler=val_data_sampler)

                                model = SynergyNetwork([n_hidden_1, n_hidden_2], X_train.shape[1], batch_norm, dropout, means, std_devs, config['norm_tanh'], V)
                                solver = Solver(optim_args={"lr": lr,
                                                            "betas": config['betas']})
                                start_epoch = 0

                                # Generate save folder
                                save_path = os.path.join(config['save_path'], 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
                                os.makedirs(save_path)

                                with open(os.path.join(save_path, 'config.txt'), 'w') as file:
                                    yaml.dump(config,file)
                                    print(yaml.dump(config))

                                solver.train(lr_decay=lr_decay,
                                             start_epoch=start_epoch,
                                             model=model,
                                             train_loader=train_data_loader,
                                             val_loader=val_data_loader,
                                             num_epochs=config['num_epochs'],
                                             log_after_iters=config['log_interval'],
                                             save_after_epochs=config['save_interval'],
                                             lr_decay_interval=lr_decay_interval,
                                             save_path=save_path,
                                             patience=config['patience']
                                             )