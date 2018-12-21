import torch
from cb18.model import SynergyNetwork
from cb18.solver import Solver
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import pickle
from cb18.utils import Dataset
import cb18.utils as utils


config = {
    'train_data_path':  '../datasets/train_data.p',
    'test_data_path': '../datasets/test_data.p',

    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'start_epoch':            100,     # Specify the number of training epochs of the existing model
    'model_path': '',
    'solver_path': '',

    'do_overfitting':       False,     # Set overfit or regular training

    #46104
    'num_train_regular':    40000,     # Number of training samples for regular training
    'num_val_regular':      6104,      # Number of validation samples for regular training
    'num_train_overfit':    256,       # Number of training samples for overfitting test runs

    'num_workers': 0,                  # Number of workers for data loading

    'mode': 'vanilla_deep_synergy',

    ## Hyperparameters ##
    'num_epochs': 1000,                # Number of epochs to train
    'batch_size': 64,
    'learning_rate': 1e-5,
    'betas': (0.9, 0.999),             # Beta coefficients for ADAM
    'lr_decay': 1,                     # Learning rate decay -> lr *= lr_decay
    'lr_decay_interval': 1500,         # Number of epochs after which to reduce the learning rate

    ## Logging ##
    'log_interval': 20,           # Number of mini-batches after which to print training loss
    'save_interval': 200,         # Number of epochs after which to save model and solver
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
print("Done.")


print("Normalizing dataset ... ", end='')
X[train_set.indices], means, std_devs = utils.normalize(X[train_set.indices], tanh=True)
X[val_set.indices], _, _ = utils.normalize(X[val_set.indices], means=means, std_devs=std_devs, tanh=True)
print("Done.")

train_data_sampler  = SubsetRandomSampler(range(len(train_set)))
val_data_sampler    = SubsetRandomSampler(range(len(val_set)))

train_data_loader   = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=train_data_sampler)
val_data_loader     = torch.utils.data.DataLoader(dataset=val_set, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=val_data_sampler)


model = SynergyNetwork(means, std_devs)
solver = Solver(optim_args={"lr": config['learning_rate'],
                            "betas": config['betas']})
start_epoch = 0

solver.train(lr_decay=config['lr_decay'],
             start_epoch=start_epoch,
             model=model,
             train_loader=train_data_loader,
             val_loader=val_data_loader,
             num_epochs=config['num_epochs'],
             log_after_iters=config['log_interval'],
             save_after_epochs=config['save_interval'],
             lr_decay_interval=config['lr_decay_interval'],
             save_path=config['save_path']
             )