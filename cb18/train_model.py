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

train_data_path = "../datasets/train_data.p"
fold_indices_path = "../datasets/fold_indices.p"
svd_load_paths = ["../datasets/svd%d.p" % i for i in range(4)]

save_path = '../saves'  # Used for saving model and solver

print("Loading train dataset ... ", end='')
with open(train_data_path, 'rb') as file:
    X, y = pickle.load(file)
print("Done.")

print("Loading fold indices ...", end='')
with open(fold_indices_path, 'rb') as file:
    fold_indices = pickle.load(file)
print("Done.")

# Training parameters
num_runs = 1             # How often to train models on the specified folds. Set this to 1 and use_bayesian to False to just
                            # train on one specific hyperparameter configuration
use_bayesian = False        # Whether to use Bayesian optimization to get hyperparameters
max_train_time_s = 18000       # Maximum training time in seconds
num_epochs = 5000           # Number of epochs to train each model
folds = range(4)            # Which folds to use for training

# Fixed hyperparameters
num_eigenvectors = 80
patience = 50  # Used for early stopping if validation performance does not improve
batch_norm = True

# Tunable hyperparameters
batch_size = 64
n_hidden_1 = 659
n_hidden_2 = 4096
learning_rate = 0.0008
dropout = 0.3
lr_decay = 1
lr_decay_interval = 1000


log_interval = None
save_interval = None

# Generate save folder for the hyperparameter search
save_path = os.path.join(save_path, 'training' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
os.makedirs(save_path)

if use_bayesian is True:
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0)

    optimizer = BayesianOptimization(
            f='',
            pbounds={'batch_size'       : (0, 0.4),
                     'n_hidden_1'       : (32, 1600),
                     'n_hidden_2'       : (500, 4000),
                     'learning_rate'    : (1e-6, 1e-4),
                     'dropout'          : (0.05, 0.35),
                     'lr_decay'         : (0.5, 0.9),
                     'lr_decay_interval': (5, 50)},
            verbose=0,
            random_state=5
    )
    logger = JSONLogger(path=os.path.join(save_path, 'logs.json'))
    optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

for i_run in range(num_runs):

    print("Testing hyperparameter configuration no. %d" % (i_run + 1))

    if use_bayesian is True:
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
    for i_fold in folds:
        n_folds += 1

        train_idx, val_idx = fold_indices[i_fold]

        X_train = X[train_idx].clone().detach()
        y_train = y[train_idx]
        X_val = X[val_idx].clone().detach()
        y_val = y[val_idx]

        print("Normalizing train data of fold %d ... " % i_fold)
        X_train, means, std_devs = utils.normalize(X_train)
        print("Done.")

        print("Loading V matrix of fold %d ... " % i_fold, end='')
        with open(svd_load_paths[i_fold], 'rb') as file:
            _, _, V = pickle.load(file)
            V = V[:, :num_eigenvectors]
        print("Done.")

        print("Projecting train data of fold %d ... " % i_fold, end='')
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

        model_save_path = os.path.join(config_save_path, "model_fold_%d.model" % i_fold)
        history_save_path = os.path.join(config_save_path, "history_fold_%d.p" % i_fold)

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

        print("Cross validation average loss after fold %d: %f\n" % (i_fold, (cum_val_loss/n_folds)))

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
