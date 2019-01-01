import pickle
import gzip
import torch
import pandas as pd
import numpy as np

input_features_load_path = "../datasets/X.p.gz"
output_data_load_path    = "../datasets/labels.csv"

test_data_save_path      = ["../datasets/test_data_fold_%d.p" % i for i in range(5)]
train_data_save_path     = ["../datasets/train_data_fold_%d.p" % i for i in range(5)]


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

# Create test data and training data for 5 folds
for i in range(5):
    # Separate the test data from the rest and save it for final evaluation
    test_idx    = np.where(fold_index == i)

    X_test  = X[test_idx]
    y_test  = synergy_scores[test_idx]

    print("Saving test dataset ... ", end='')
    with open(test_data_save_path[i], 'wb') as saveFile:
        pickle.dump((X_test, y_test), saveFile)
    print("Done.")


    # Get the training data and save it for later training
    train_idx = np.where(fold_index != i)

    X_train = X[train_idx]
    y_train = synergy_scores[train_idx]

    print("Saving training dataset ... ", end='')
    with open(train_data_save_path[i], 'wb') as saveFile:
        pickle.dump((X_train, y_train), saveFile)
    print("Done.")


