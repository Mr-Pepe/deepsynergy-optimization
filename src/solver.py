from random import shuffle
import numpy as np
import datetime
import torch
from torch.autograd import Variable
import os
import pickle

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.MSELoss):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func

        self.lr = self.optim_args['lr']

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.val_loss_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_after_iters=1, save_after_epochs=1, start_epoch=0,
              lr_decay=1, lr_decay_interval=1, save_path='../saves/train', ):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        optim = self.optim(model.parameters(), **self.optim_args)

        if start_epoch == 0:
            self._reset_histories()

        iter_per_epoch = len(train_loader)

        # Exponentially filtered training loss
        loss_avg = 0

        # Generate save folder
        save_path = os.path.join(save_path, 'train' + datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
        os.makedirs(save_path)

        # Calculate the total number of minibatches for the training procedure
        n_iters = num_epochs*iter_per_epoch
        i_iter = 0


        print('START TRAIN.')

        # Do the training here
        for i_epoch in range(num_epochs):

            i_epoch += start_epoch

            # Set model to train mode
            model.train()

            for i_iter_in_epoch, batch in enumerate(train_loader):
                i_iter += 1

                X = batch['x']
                y = batch['y']

                # Transfer batch to GPU if available
                X = X.to(device)
                y = y.to(device)

                # Forward pass
                pred = model(X)

                # Calculate loss
                criterion = torch.nn.MSELoss()
                loss = criterion(pred, y)

                # Packpropagate and update weights
                model.zero_grad()
                loss.backward()
                optim.step()

                # Save loss to history
                self.train_loss_history.append(loss.item())
                loss_avg = 99/100*loss_avg + 1/100*loss.item()

                if i_iter%log_after_iters == 0:
                    print("Iteration " + str(i_iter) + "/" + str(n_iters) + "   Train loss: " + str(loss.item()) + "   Avg: " + str(loss_avg))

            # Save model and solver
            if (i_epoch+1)%save_after_epochs == 0:
                model.save(save_path + '/model' + str(i_epoch + 1))
                self.save(save_path + '/solver' + str(i_epoch + 1))
                model.to(device)

            # Validate model
            print("Validate model after epoch " + str(i_epoch+1))

            # Set model to evaluation mode
            model.eval()


            num_val_batches = 0


            for i, batch in enumerate(val_loader):
                num_val_batches += 1

                X = batch['x']
                y = batch['y']

                # Transfer batch to GPU if available
                X = X.to(device)
                y = y.to(device)

                # Forward pass
                pred = model(X)

                # Calculate loss
                criterion = torch.nn.MSELoss()
                loss = criterion(pred, y)

                self.val_loss_history.append(loss.item())

                print("Epoch " + str(i_epoch) + '/' + str(num_epochs) + '   Val loss: '+ str(loss.item()))

            # Update learning rate
            if i_epoch%lr_decay_interval == 0:
                self.lr *= lr_decay
                print("Learning rate: " + str(self.lr))
                for i, _ in enumerate(optim.param_groups):
                    optim.param_groups[i]['lr'] = self.lr

        # Save model and solver after training
        model.save(save_path + '/model' + str(i_epoch + 1))
        self.save(save_path + '/solver' + str(i_epoch + 1))

        print('FINISH.')


    def save(self, path):
        print('Saving solver... %s' % path)
        pickle.dump(self, open(path, 'wb'))
