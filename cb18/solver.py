import datetime
import torch
import os
import pickle
import time

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
        self.lr_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_after_iters=1, save_after_epochs=1, start_epoch=0,
              lr_decay=1, lr_decay_interval=1, save_path='../saves/train', patience=None, max_train_time_s=300000):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("Using device: " + device.type)

        model.to(device)

        optim = self.optim(model.parameters(), **self.optim_args)

        if start_epoch == 0:
            self._reset_histories()

        iter_per_epoch = len(train_loader)

        # Exponentially filtered training loss
        train_loss_avg = 0
        best_val_loss  = 0
        patience_counter = 0
        val_loss_avg = 0


        # Calculate the total number of minibatches for the training procedure
        n_iters = num_epochs*iter_per_epoch
        i_iter = 0

        t_start_training = time.time()

        print('START TRAINING.')

        # Do the training here
        for i_epoch in range(num_epochs):
            t_start_epoch = time.time()

            i_epoch += start_epoch

            # Set model to train mode
            model.train()

            for i_iter_in_epoch, batch in enumerate(train_loader):
                t_start_iter = time.time()

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
                loss = criterion(pred[:,0], y)

                # Packpropagate and update weights
                model.zero_grad()
                loss.backward()
                optim.step()

                # Save loss to history
                smooth_window_train = 15

                self.train_loss_history.append(loss.item())
                train_loss_avg = (smooth_window_train-1)/smooth_window_train*train_loss_avg + 1/smooth_window_train*loss.item()

                # if i_iter%log_after_iters == 0:
                #     print("Iteration " + str(i_iter) + "/" + str(n_iters) + "   Train loss: " + "{0:.3f}".format(loss.item()) + "   Avg: " + "{0:.3f}".format(loss_avg) + " - " + str(int((time.time()-t_start_iter)*1000)) + "ms")

            # Save model and solver
            if (i_epoch+1)%save_after_epochs == 0:
                model.save(save_path + '/model' + str(i_epoch + 1))
                self.save(save_path + '/solver' + str(i_epoch + 1))
                model.to(device)


            # Set model to evaluation mode
            model.eval()

            num_val_batches = 0
            val_loss = 0

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
                loss = criterion(pred[:,0], y)

                val_loss += loss.item()

            val_loss /= num_val_batches
            self.val_loss_history.append(val_loss)

            smooth_window_val = 10

            self.train_loss_history.append(loss.item())
            if val_loss_avg == 0:
                val_loss_avg = val_loss
            else:
                val_loss_avg = (smooth_window_val - 1) / smooth_window_val * val_loss_avg + 1 / smooth_window_val * val_loss


            print("Epoch " + str(i_epoch+1) + '/' + "{0:.3f}".format(num_epochs) + '   Val loss: '+ "{0:.3f}".format(loss.item()) + "   - Avg: " + "{0:.3f}".format(val_loss_avg) + "   - " + str(int((time.time()-t_start_epoch)*1000)) + "ms" )

            self.lr_history.append(self.lr)

            # Update learning rate
            if (i_epoch+1)%lr_decay_interval == 0:
                self.lr *= lr_decay
                print("Learning rate: " + str(self.lr))
                for i, _ in enumerate(optim.param_groups):
                    optim.param_groups[i]['lr'] = self.lr

            # Early stopping
            if patience is not None:
                if best_val_loss == 0 or val_loss_avg < best_val_loss:
                    best_val_loss = val_loss_avg
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter > patience:
                    stop_training = True
                    print("Early stopping after " + str(i_epoch+1) + " epochs.")
                    break

            # Stop if training time is over
            if time.time()-t_start_training > max_train_time_s:
                print("Training time is over.")
                break

        # Save model and solver after training
        model.save(save_path + '/model' + str(i_epoch + 1))
        self.save(save_path + '/solver' + str(i_epoch + 1))

        print('FINISH.\n\n\n')

        # Return last smoothened validation loss
        return val_loss_avg


    def save(self, path):
        print('Saving solver... %s' % path)
        pickle.dump(self, open(path, 'wb'))
