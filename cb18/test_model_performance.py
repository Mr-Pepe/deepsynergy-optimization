import cb18.utils as utils
import cb18.model
import pickle
import torch
import os
import os.path
import matplotlib.pyplot as plt

test_data_paths  = ["../datasets/test_data_fold_%d.p" % i for i in range(5)]
model_dir_path  = '../saves/training20190103093146'

for i_test_fold in range(5):

    # Load test dataset
    print("Loading test dataset for test fold %d ... " %i_test_fold, end='')
    with open(test_data_paths[i_test_fold], 'rb') as file:
        X, y = pickle.load(file)
        dataset = utils.Dataset(X, y)
    print("Done.")

    # Find the corresponding models on the path
    model_paths = []
    for dirpath, dirnames, filenames in os.walk(model_dir_path):
        for filename in [f for f in filenames if f.find('model_test_fold_%d' %i_test_fold) != -1]:
            model_paths.append(os.path.join(dirpath, filename))


    overall_pred_scores = None
    n_overall_pred_scores = 0
    overall_mse = 0
    mse_combined_history = []
    mse_history = []

    for path in model_paths:

        # print("Loading model " + path + " ... ", end='')
        model = torch.load(path)
        model.eval()
        # print("Done.")

        # print("Evaluating model ... ", end='')
        pred_scores = model(X)
        del model
        pred_scores = pred_scores.detach()
        # print("Done.")

        if overall_pred_scores is None:
            overall_pred_scores = torch.t(pred_scores)
            n_overall_pred_scores = 1
        else:
            overall_pred_scores = n_overall_pred_scores / (n_overall_pred_scores + 1) * overall_pred_scores + torch.t(
                pred_scores) / (n_overall_pred_scores + 1)
            n_overall_pred_scores += 1
            # overall_pred_scores = torch.cat((overall_pred_scores, torch.t(pred_scores)), 0)

        mse = pred_scores[:, 0] - y
        mse = mse.pow(2)
        mse = mse.sum() / len(mse)

        mse_history.append(mse.item())

        mse_combined = overall_pred_scores.mean(dim=0) - y
        mse_combined = mse_combined.pow(2)
        mse_combined = mse_combined.sum() / len(mse_combined)

        mse_combined_history.append(mse_combined.item())

        print("Test score this model: " + "{0:.3f}".format(mse.item()) + "  Combined: " + "{0:.3f}".format(
            mse_combined.item()))

    plt.scatter(y.numpy(), overall_pred_scores.mean(dim=0).numpy())