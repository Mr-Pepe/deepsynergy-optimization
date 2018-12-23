"""ClassificationCNN"""
import torch
import torch.nn as nn
import cb18.utils as utils


class SynergyNetwork(nn.Module):
    

    def __init__(self, n_hidden, input_dim, batch_norm, dropout, means=None, std_devs=None, tanh=None, V=None):

        super(SynergyNetwork, self).__init__()

        self.in_norm = nn.BatchNorm1d(input_dim)
        self.layer1 = FCLayer(input_dim, n_hidden[0],dropout=dropout, batchnorm=batch_norm)
        self.layer2 = FCLayer(n_hidden[0], n_hidden[1], dropout=dropout, batchnorm=batch_norm)
        self.outlayer = nn.Linear(n_hidden[1], 1)

        if means is None or std_devs is None or tanh is None:
            raise("Need means or std devs or tanh for model initialization")
        else:
            self.means = means
            self.std_devs = std_devs
            self.tanh = tanh
            self.V = V



    def forward(self, input):

        if self.training is False:
            input, _, _ = utils.normalize(input.cpu(), means=self.means, std_devs=self.std_devs, tanh=self.tanh)
            input = torch.matmul(input.cpu(), self.V)

            if self.is_cuda:
                input = input.cuda()

        out = self.in_norm(input)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.outlayer(out)

        return out

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self.cpu(), path)



class FCLayer(nn.Module):

    def __init__(self, input, output, batchnorm=False, dropout=0):
        super(FCLayer, self).__init__()

        self.use_batchnorm = batchnorm

        self.linear = nn.Linear(input, output)
        self.batchnorm = nn.BatchNorm1d(output)
        self.dropout = nn.Dropout(p=dropout)
        self.relu   = nn.ReLU()

    def forward(self, input):
        out = self.linear(input)

        if self.use_batchnorm is True:
            out = self.batchnorm(out)

        out = self.dropout(out)
        out = self.relu(out)

        return out