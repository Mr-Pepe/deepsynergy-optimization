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
import numpy as np
import pandas as pd

# Train model ensembles for all test folds
train_data_path     = ["../datasets/train_data_fold_%d.p" % i for i in range(5)]
svd_load_paths      = ["../datasets/svd_fold_%d.p" % i for i in range(5)]
fold_index_path     = "../datasets/labels.csv"

save_path = '../saves'  # Used for saving model and solver

# Training parameters
max_train_time_s = 180000  # Maximum training time in seconds for each model
num_epochs = 5000  # Number of epochs to train each model

# Fixed hyperparameters
n_train_folds = 10
num_eigenvectors = 107
patience = 50  # Used for early stopping if validation performance does not improve
batch_norm = True
use_given_folds = True

log_interval = None
save_interval = None

# Generate save folder for the hyperparameter search
save_path = os.path.join(save_path, 'training' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
os.makedirs(save_path)

for i_test_fold in range(5):

    print("Loading train dataset for fold %d ... " % i_test_fold, end='')
    with open(train_data_path[i_test_fold], 'rb') as file:
        X, y = pickle.load(file)
    print("Done.")

    print("Loading V matrix of test fold %d ... " % i_test_fold, end='')
    with open(svd_load_paths[i_test_fold], 'rb') as file:
        _, V = pickle.load(file)
        V = V[:, :num_eigenvectors]
    print("Done.")

    if use_given_folds:
        labels = pd.read_csv(fold_index_path, index_col=0)
        labels = pd.concat([labels, labels])
        fold_indices = labels.values[:, 4].astype('int')
        fold_indices = fold_indices[np.where(fold_indices != i_test_fold)]
        n_train_folds = 5
    else:
        np.random.seed(0)
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        fold_indices = np.array_split(indices, n_train_folds)

    ## DeepSynergy parameters
    # batch_size = 64
    # n_hidden_1 = 8192
    # n_hidden_2 = 4096
    # learning_rate = 1e-5
    # dropout = 0.5
    # lr_decay = 1
    # lr_decay_interval = 5000

    batch_size = 2048
    n_hidden_1 = 512
    n_hidden_2 = 256
    learning_rate = 0.001
    dropout = 0.3
    lr_decay = 1
    lr_decay_interval = 5000

    with open(os.path.join(save_path, 'config.txt'), 'w') as file:
        print(yaml.dump({'num_eigenvectors': num_eigenvectors, 'n_hidden_1': n_hidden_1, 'n_hidden_2': n_hidden_2, 'batch_norm': batch_norm,
                         'dropout'   : dropout, 'batch_size': batch_size, 'learning_rate': learning_rate,
                         'lr_decay'  : lr_decay, 'lr_decay_interval': lr_decay_interval}))
        yaml.dump({'num_eigenvectors': num_eigenvectors, 'n_hidden_1': n_hidden_1, 'n_hidden_2': n_hidden_2, 'batch_norm': batch_norm,
                   'dropout'   : dropout, 'batch_size': batch_size, 'learning_rate': learning_rate,
                   'lr_decay'  : lr_decay, 'lr_decay_interval': lr_decay_interval}, file)

    cum_val_loss = 0
    n_folds = 0


    # Do the training and cross validation
    for i_train_fold in range(n_train_folds):

        if use_given_folds:
            if i_train_fold == i_test_fold:
                continue
            else:
                train_idx = np.where(fold_indices != i_train_fold)
                val_idx = np.where(fold_indices == i_train_fold)
        else:
            train_idx   = np.delete(indices, np.where(np.isin(indices, fold_indices[i_train_fold])))
            val_idx     = fold_indices[i_train_fold]

        n_folds += 1

        X_train = X[train_idx].clone().detach()
        y_train = y[train_idx]
        X_val = X[val_idx].clone().detach()
        y_val = y[val_idx]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_train.to(device)
        y_train.to(device)
        X_val.to(device)
        y_val.to(device)

        print("Normalize train data of test fold %d train fold %d ... " % (i_test_fold, i_train_fold))
        X_train, means, std_devs = utils.normalize(X_train, tanh=False)
        print("Done.")

        print("Projecting train data of train fold %d ... " % i_train_fold, end='')
        X_train = torch.matmul(X_train, V)
        print("Done. ")

        train_set = Dataset(X_train, y_train)
        val_set = Dataset(X_val, y_val)

        train_data_sampler = SubsetRandomSampler(range(len(train_set)))
        val_data_sampler = SubsetRandomSampler(range(len(val_set)))

        train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size,
                                                        sampler=train_data_sampler)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, sampler=val_data_sampler)

        model = SynergyNetwork([n_hidden_1, n_hidden_2], X_train.shape[1], batch_norm, dropout, means, std_devs, V)
        solver = Solver(optim_args={"lr": learning_rate})
        start_epoch = 1

        model_save_path = os.path.join(save_path, "model_test_fold_%d_train_fold_%d.model" % (i_test_fold, i_train_fold))
        history_save_path = os.path.join(save_path, "history_test_fold_%d_train_fold_%d.p" % (i_test_fold, i_train_fold))

        val_loss = solver.train(lr_decay=lr_decay,
                                start_epoch=start_epoch,
                                model=model,
                                train_loader=train_data_loader,
                                val_loader=val_data_loader,
                                num_epochs=num_epochs,
                                log_after_iters=log_interval,
                                save_after_epochs=save_interval,
                                lr_decay_interval=lr_decay_interval,
                                model_save_path=model_save_path,
                                history_save_path=history_save_path,
                                patience=patience,
                                max_train_time_s=max_train_time_s
                                )

        cum_val_loss += val_loss

        print("Cross validation average loss after fold %d: %f\n" % (i_train_fold, (cum_val_loss / n_folds)))

        del model
        del solver
        del X_train
        del X_val
        del y_train
        del y_val

