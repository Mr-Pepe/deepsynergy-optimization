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
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
import numpy as np
import pandas as pd

# Specify the fold on which to perform the hyperparameter search
train_data_path = "../datasets/train_data_fold_0.p"
svd_load_path = "../datasets/svd_fold_0.p"
fold_index_path     = "../datasets/labels.csv"

save_path = '../saves'  # Used for saving model and solver

print("Loading train dataset ... ", end='')
with open(train_data_path, 'rb') as file:
    X, y = pickle.load(file)
print("Done.")


# Training parameters
num_runs = 100                # How often to train model
max_train_time_s = 18000    # Maximum training time in seconds
num_epochs = 5000           # Number of epochs to train each model

# Fixed hyperparameters
num_eigenvectors = 80
patience = 50  # Used for early stopping if validation performance does not improve
batch_norm = True

log_interval = None
save_interval = None

# Generate save folder for the hyperparameter search
save_path = os.path.join(save_path, 'hypersearch' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
os.makedirs(save_path)

print("Loading V matrix... ", end='')
with open(svd_load_path, 'rb') as file:
    _, V = pickle.load(file)
    V = V[:, :num_eigenvectors]
print("Done.")

utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0)

optimizer = BayesianOptimization(
        f='',
        pbounds={'batch_size'       : (0, 0.4),
                 'n_hidden_1'       : (256, 4096),
                 'n_hidden_2'       : (256, 4096),
                 'learning_rate'    : (1e-5, 1e-5),
                 'dropout'          : (0.05, 0.5),
                 'lr_decay'         : (1, 1),
                 'lr_decay_interval': (100, 100)},
        verbose=0,
        random_state=5
)
logger = JSONLogger(path=os.path.join(save_path, 'logs.json'))
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

labels = pd.read_csv(fold_index_path, index_col=0)
labels = pd.concat([labels, labels])
fold_indeces = labels.values[:, 4].astype('int')
fold_indeces = fold_indeces[np.where(fold_indeces != 0)]

for i_run in range(num_runs):

    print("Testing hyperparameter configuration no. %d" % (i_run + 1))

    next = optimizer.suggest(utility)

    if next['batch_size'] > 0.5:
        batch_size = 512
    else:
        batch_size = 64

    n_hidden_1 = int(next['n_hidden_1'])
    n_hidden_2 = int(next['n_hidden_2'])
    learning_rate = float(next['learning_rate'])
    dropout = float(next['dropout'])
    lr_decay = float(next['lr_decay'])
    lr_decay_interval = int(next['lr_decay_interval'])

    # Generate save folder for the config
    config_save_path = os.path.join(save_path, "config%d" % i_run)
    os.makedirs(config_save_path)

    with open(os.path.join(config_save_path, 'config.txt'), 'w') as file:
        print(yaml.dump({'n_hidden_1': n_hidden_1, 'n_hidden_2': n_hidden_2, 'batch_norm': batch_norm,
                         'dropout'   : dropout, 'batch_size': batch_size, 'learning_rate': learning_rate,
                         'lr_decay'  : lr_decay, 'lr_decay_interval': lr_decay_interval}))
        yaml.dump({'n_hidden_1': n_hidden_1, 'n_hidden_2': n_hidden_2, 'batch_norm': batch_norm,
                   'dropout'   : dropout, 'batch_size': batch_size, 'learning_rate': learning_rate,
                   'lr_decay'  : lr_decay, 'lr_decay_interval': lr_decay_interval}, file)



    cum_val_loss = 0
    n_folds = 0

    # Do the training and cross validation
    for i_train_fold in range(1, 5):
        n_folds += 1

        train_idx = np.where(fold_indeces != i_train_fold)
        val_idx = np.where(fold_indeces == i_train_fold)

        X_train = X[train_idx].clone().detach()
        y_train = y[train_idx]
        X_val = X[val_idx].clone().detach()
        y_val = y[val_idx]

        print("Normalizing train data of train fold %d ... " % i_train_fold)
        X_train, means, std_devs = utils.normalize(X_train)
        print("Done.")

        print("Projecting train data of train fold %d ... " % i_train_fold, end='')
        X_train = torch.matmul(X_train, V)
        print("Done. ")

        train_set = Dataset(X_train, y_train)
        val_set = Dataset(X_val, y_val)

        train_data_sampler = SubsetRandomSampler(range(len(train_set)))
        val_data_sampler = SubsetRandomSampler(range(len(val_set)))

        train_data_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, sampler=train_data_sampler)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size, sampler=val_data_sampler)

        model = SynergyNetwork([n_hidden_1, n_hidden_2], X_train.shape[1], batch_norm, dropout, means, std_devs, V)
        solver = Solver(optim_args={"lr": learning_rate})
        start_epoch = 1

        model_save_path = os.path.join(config_save_path, "model_fold_%d.model" % i_train_fold)
        history_save_path = os.path.join(config_save_path, "history_fold_%d.p" % i_train_fold)

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
        del V

    # Use negative val_loss because the optimizer maximizes
    optimizer.register(params=next, target=-(cum_val_loss/n_folds))
    print(optimizer.max)
