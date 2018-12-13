import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import pickle

class Dataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(X)

    def __getitem__(self, idx):
        sample = {'x': self.X[idx], 'y': self.y[idx]}

        return sample


def eval_model(model, images):
    plt.interactive(False)
    for image in images:
        to_pil = transforms.ToPILImage()

        plt.subplot(1,3,1)
        plt.imshow(to_pil(image))

        restored = model(torch.unsqueeze(image, 0))

        plt.subplot(1,3,2)
        plt.imshow(to_pil(restored[0]))

        plt.subplot(1, 3, 3)
        plt.imshow(to_pil(restored[0]-image))
        plt.show(block=True)


def show_solver_history(path):
    solver = pickle.load(open(path, 'rb'))

    loss = np.array(solver.train_loss_history)
    psnr = np.array(solver.psnr_history)
    num_per_subtask = np.array(solver.num_per_subtask_history)

    plt.plot(loss[60000:])
    plt.plot(np.convolve(loss[60000:], np.ones((1000,)) / 1000, mode='valid'))
    plt.xlabel("Iterations")
    plt.ylabel("Train loss")
    plt.show()

    plt.plot(psnr[:, 0])
    plt.plot(psnr[:, 1])
    plt.plot(psnr[:, 2])
    plt.plot(psnr[:, 3])
    plt.plot(psnr[:, 4])
    plt.xlabel("Iterations")
    plt.ylabel("PSNR")
    plt.show()

    batchsize = num_per_subtask.sum(axis=1)

    plt.plot(num_per_subtask[:, 0] / batchsize * 100)
    plt.plot(num_per_subtask[:, 1] / batchsize * 100)
    plt.plot(num_per_subtask[:, 2] / batchsize * 100)
    plt.plot(num_per_subtask[:, 3] / batchsize * 100)
    plt.plot(num_per_subtask[:, 4] / batchsize * 100)
    plt.xlabel("Epochs")
    plt.ylabel("Samples per subtask")
    plt.axes().xaxis.label.set_fontsize(20)
    plt.axes().yaxis.label.set_fontsize(20)
    [x.set_fontsize(12) for x in plt.axes().get_xticklabels()]
    [x.set_fontsize(12) for x in plt.axes().get_yticklabels()]
    plt.show()

    print(loss[5000:7000].mean())
    print(loss[10000:12000].mean())
    print(loss[15000:17000].mean())
    print(loss[20000:23000].mean())
    print(loss[25000:28000].mean())
    print(loss[300000:305000].mean())
