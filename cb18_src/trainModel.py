import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from cb18_src.model import SynergyNetwork
from cb18_src.solver import Solver
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import pickle
from cb18_src.utils import Dataset


config = {

    'input_features_path':        '../datasets/x_norm_tanh.p',
    'synergy_scores_path': '../datasets/synergy_scores.p',

    'continue_training':   False,      # Specify whether to continue training with an existing model and solver
    'start_epoch':            100,             # Specify the number of training epochs of the existing model
    'model_path': '',
    'solver_path': '',

    'do_overfitting':       False,            # Set overfit or regular training

    #46104
    'num_train_regular':    40000,     # Number of training samples for regular training
    'num_val_regular':      6104,       # Number of validation samples for regular training
    'num_train_overfit':    256,       # Number of training samples for overfitting test runs

    'num_workers': 0,                   # Number of workers for data loading

    'mode': 'vanilla_deep_synergy',

    ## Hyperparameters ##
    'num_epochs': 1000,                  # Number of epochs to train
    'batch_size': 64,
    'learning_rate': 1e-5,
    'betas': (0.9, 0.999),              # Beta coefficients for ADAM
    'lr_decay': 1,                      # Learning rate decay -> lr *= lr_decay
    'lr_decay_interval': 1500,             # Number of epochs after which to reduce the learning rate

    ## Logging ##
    'log_interval': 1,           # Number of mini-batches after which to print training loss
    'save_interval': 100,         # Number of epochs after which to save model and solver
    'save_path': '../saves'
}

input_features_file = open(config['input_features_path'], 'rb')
input_features = pickle.load(input_features_file)
input_features_file.close()

synergy_scores_file = open(config['synergy_scores_path'], 'rb')
synergy_scores = pickle.load(synergy_scores_file)
synergy_scores_file.close()

dataset = Dataset(input_features, synergy_scores)

if config['do_overfitting']:
    train_data_sampler  = SequentialSampler(range(config['num_train_overfit']))
    val_data_sampler    = SequentialSampler(range(config['num_train_overfit']))

else:
    if config['num_train_regular']+config['num_val_regular'] > len(dataset):
        raise Exception('Trying to use more samples for training and validation than are available.')
    else:
        train_data_sampler  = SubsetRandomSampler(range(config['num_train_regular']))
        val_data_sampler    = SubsetRandomSampler(range(config['num_val_regular']))

        [train_set, val_set] = torch.utils.data.random_split(dataset, [config['num_train_regular'], config['num_val_regular']])

        train_data_loader   = torch.utils.data.DataLoader(dataset=train_set, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=train_data_sampler)
        val_data_loader     = torch.utils.data.DataLoader(dataset=val_set, batch_size=config['batch_size'], num_workers=config['num_workers'], sampler=val_data_sampler)


if config['continue_training']:
    model = torch.load(config['model_path'])
    solver = pickle.load(open(config['solver_path'], 'rb'))
    start_epoch = config['start_epoch']
else:
    model = SynergyNetwork()
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