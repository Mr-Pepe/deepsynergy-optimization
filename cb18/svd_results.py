import pickle
import matplotlib.pyplot as plt
import matplotlib
import torch

svd_load_paths      = ["../datasets/svd_fold_%d.p" % i for i in range(5)]

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

for i_test_fold in range(5):

    print("Loading S for test fold %d ... " % i_test_fold, end='')
    with open(svd_load_paths[i_test_fold], 'rb') as file:
        S, _ = pickle.load(file)
    print("Done.")


    # Plot variance coverage
    var = torch.cumsum(S.pow(2) / S.pow(2).sum(), 0)

    fig = plt.figure()

    plt.plot(var[:200].numpy())
    plt.title("Variance Coverage")
    plt.xlabel("Ordered Eigenvectors")
    plt.ylabel("Cumulative covered Variance")
    plt.show()

    fig.savefig('covered_variance.pdf', format='pdf')




