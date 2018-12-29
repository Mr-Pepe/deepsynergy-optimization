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
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events

config = {
    'train_data_path':  '../datasets/train_data.p',
    'test_data_path': '../datasets/test_data.p',

    'num_eigenvectors': 80,
    'norm_tanh': False,
    'patience': 50,
    'n_bayes_steps': 100,

    ## Hyperparameters ##
    'num_epochs':   5000,                # Number of epochs to train
    'batch_norm':   True,
    'betas': (0.9, 0.999),             # Beta coefficients for ADAM
    'lr_decay': 1,                     # Learning rate decay -> lr *= lr_decay
    'lr_decay_interval': 1500,         # Number of epochs after which to reduce the learning rate

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


utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0)

optimizer = BayesianOptimization(
    f='',
    pbounds={'batch_size':   (0, 0.4),
            'n_hidden_1':   (32, 1600),
            'n_hidden_2':   (2000, 4000),
            'learning_rate': (1e-6, 1e-4),
            'dropout':      (0.05, 0.35),
             'lr_decay':    (0.5, 0.9),
             'lr_decay_interval': (10, 100)},
    verbose=0,
    random_state=5
)

# Generate save folder for the hyperparameter search
save_path = os.path.join(config['save_path'], 'search' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
os.makedirs(save_path)

logger = JSONLogger(path=os.path.join(save_path, 'logs.json'))
optimizer.subscribe(Events.OPTMIZATION_STEP, logger)

for i in range(config['n_bayes_steps']):


    next = optimizer.suggest(utility)

    if next['batch_size'] > 0.5:
        batch_size = 512
    else:
        batch_size = 64

    n_hidden_1 = int(next['n_hidden_1'])
    n_hidden_2 = int(next['n_hidden_2'])
    learning_rate = float(next['learning_rate'])
    dropout = float(next['dropout'])
    batch_norm = config['batch_norm']
    lr_decay = float(next['lr_decay'])
    lr_decay_interval = int(next['lr_decay_interval'])

    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=batch_size,
                                                    sampler=train_data_sampler)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=batch_size,
                                                    sampler=val_data_sampler)


    model = SynergyNetwork([n_hidden_1, n_hidden_2], X_train.shape[1], batch_norm, dropout,
                           means, std_devs, config['norm_tanh'], V)
    solver = Solver(optim_args={"lr": learning_rate,
                                "betas": config['betas']})
    start_epoch = 0

    # Generate save folder
    train_save_path = os.path.join(save_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    os.makedirs(train_save_path)

    with open(os.path.join(train_save_path, 'config.txt'), 'w') as file:
        print(yaml.dump(
            {'n_hidden_1':n_hidden_1, 'n_hidden_2': n_hidden_2, 'batch_norm': batch_norm,
                   'dropout': dropout, 'batch_size': batch_size, 'learning_rate': learning_rate,
                   'lr_decay': lr_decay, 'lr_decay_interval': lr_decay_interval}))
        yaml.dump({'n_hidden_1':n_hidden_1, 'n_hidden_2': n_hidden_2, 'batch_norm': batch_norm,
                   'dropout': dropout, 'batch_size': batch_size, 'learning_rate': learning_rate,
                   'lr_decay': lr_decay, 'lr_decay_interval': lr_decay_interval},file)


    val_loss = solver.train(lr_decay=lr_decay,
                             start_epoch=start_epoch,
                             model=model,
                             train_loader=train_data_loader,
                             val_loader=val_data_loader,
                             num_epochs=config['num_epochs'],
                             log_after_iters=config['log_interval'],
                             save_after_epochs=config['save_interval'],
                             lr_decay_interval=lr_decay_interval,
                             save_path=train_save_path,
                             patience=config['patience'],
                             max_train_time_s = 7200
                            )

    # Use negative val_loss because the optimizer maximizes
    optimizer.register(params=next, target=-val_loss)
    print(val_loss, next)
    print(optimizer.max)