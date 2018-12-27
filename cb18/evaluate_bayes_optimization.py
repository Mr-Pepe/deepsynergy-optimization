from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from bayes_opt.util import load_logs
import os
import torch
import pickle
from cb18.utils import Dataset
import numpy as np
import matplotlib.pyplot as plt

log_path = '../saves/search20181224210125/logs.json'
train_data_path = '../datasets/train_data.p'
model_dir_path  = '../saves/search20181224210125'


model_paths = []
for dirpath, dirnames, filenames in os.walk(model_dir_path):
    for filename in [f for f in filenames if f.find('model') != -1]:
        model_paths.append(os.path.join(dirpath, filename))

model_paths = sorted(model_paths, key=str.lower)


optimizer = BayesianOptimization(
    f='',
    pbounds={'batch_size':   (0, 1),
            'n_hidden_1':   (32, 4096),
            'n_hidden_2':   (32, 4096),
            'learning_rate': (1e-5, 1e-3),
            'dropout':      (0, 1)},
    verbose=0,
    random_state=5
)

load_logs(optimizer, logs=[log_path])

print("Loading train dataset ... ", end='')
with open(train_data_path, 'rb') as train_data_file:
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

sorted_idx = optimizer.space.target.argsort()

n_best_models = 92

overall_pred_scores = None
n_overall_pred_scores = 0
overall_mse = 0
mse_combined_history = []
mse_history = []

for i in range(n_best_models):
    path = model_paths[sorted_idx[i]]

    # Load model
    model = torch.load(path)
    model.eval()

    # Evaluate model
    pred_scores = model(X_val)
    del model
    pred_scores = pred_scores.detach()

    if overall_pred_scores is None:
        overall_pred_scores = torch.t(pred_scores)
        n_overall_pred_scores = 1
    else:
        overall_pred_scores = n_overall_pred_scores/(n_overall_pred_scores+1)*overall_pred_scores + torch.t(pred_scores)/(n_overall_pred_scores+1)
        n_overall_pred_scores += 1
        # overall_pred_scores = torch.cat((overall_pred_scores, torch.t(pred_scores)), 0)

    mse = pred_scores[:, 0] - y_val
    mse = mse.pow(2)
    mse = mse.sum() / len(mse)

    mse_history.append(mse.item())


    mse_combined = overall_pred_scores.mean(dim=0) - y_val
    mse_combined = mse_combined.pow(2)
    mse_combined = mse_combined.sum()/ len(mse_combined)

    mse_combined_history.append(mse_combined.item())

    print("Val score this model: " + "{0:.3f}".format(mse.item()) + "  Combined: " + "{0:.3f}".format(mse_combined.item()))


plt.plot(mse_history)
plt.plot(mse_combined_history)
plt.show()

with open("../doc/single_vs_ensemble.p", 'wb') as file:
    pickle.dump((mse_history, mse_combined_history), file)

print("Done.")