"""ClassificationCNN"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class SynergyNetwork(nn.Module):
    

    def __init__(self, input_dim=12758):

        super(SynergyNetwork, self).__init__()

        n_hidden = (8182, 4096)

        self.layer1 = FCLayer(input_dim, n_hidden[0],dropout=0.2)
        self.layer2 = FCLayer(n_hidden[0], n_hidden[1], dropout=0.5)
        self.outlayer = nn.Linear(n_hidden[1], 1)


    def forward(self, input):

        out = self.layer1(input)
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