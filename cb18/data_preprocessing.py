import pickle
import gzip
import torch
import pandas as pd
import numpy as np

input_features_load_path = "../datasets/X.p.gz"
output_data_load_path    = "../datasets/labels.csv"

test_data_save_path    = "../datasets/test_data.p"
train_data_save_path   = "../datasets/train_data.p"
fold_indices_save_path = "../datasets/fold_indices.p"


## Load the input data
print("Loading input features ... ", end='')
with gzip.open(input_features_load_path) as loadFile:
    X = pickle.load(loadFile)
    X = torch.from_numpy(X)
    X = X.float()
print("Done.")

## Load the output data
print("Loading labels ... ", end='')
labels = pd.read_csv(output_data_load_path, index_col=0)
labels = pd.concat([labels, labels])
print("Done!")

synergy_scores = labels.values[:,3].astype('float')
synergy_scores = torch.from_numpy(synergy_scores)
synergy_scores = synergy_scores.float()


# Split the data according to the folds given by Preuer et al.
print("Splitting data ... ")
fold_index = labels.values[:, 4].astype('int')

# Separate the test data from the rest and save it for final evaluation
test_idx    = np.where(fold_index == 0)

X_test  = X[test_idx]
y_test  = synergy_scores[test_idx]

print("Saving test dataset ... ", end='')
with open(test_data_save_path, 'wb') as saveFile:
    pickle.dump((X_test, y_test), saveFile)
print("Done.")


# Get the training data and save it for later training
train_idx = np.where(fold_index != 0)

X_train = X[train_idx]
y_train = synergy_scores[train_idx]

print("Saving training dataset ... ", end='')
with open(train_data_save_path, 'wb') as saveFile:
    pickle.dump((X_train, y_train), saveFile)
print("Done.")

# Create folds for the fold indices 1 ... 4 where in each fold one index is held out as validation set
# The indices for each fold are stored for later use during training
fold_indices = []

# Indices must index the train dataset and not the whole dataset including the test data
fold_index = fold_index[train_idx]


for i in range(1,5):
    train_idx = np.where(fold_index != i)
    val_index = np.where(fold_index == i)

    fold_indices.append((train_idx[0], val_index[0]))


print("Saving indices for the folds ...", end='')
with open(fold_indices_save_path, 'wb') as saveFile:
    pickle.dump(fold_indices, saveFile)
print("Done.")



