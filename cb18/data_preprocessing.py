import pickle
import gzip
import torch
import pandas as pd
import numpy as np


## Load the input data
print("Loading input features")
loadFile = gzip.open("../datasets/X.p.gz")

X = pickle.load(loadFile)
loadFile.close()

X = torch.from_numpy(X)
X = X.float()

## Load the output data
print("Loading labels ... ", end='')
labels = pd.read_csv("../datasets/labels.csv", index_col=0)
labels = pd.concat([labels, labels])
print("Done!")

print("Split data ... ", end='')
synergy_scores = labels.values[:,3].astype('float')
synergy_scores = torch.from_numpy(synergy_scores)
synergy_scores = synergy_scores.float()
print("Done!")

idx = labels.values[:,4].astype('int')
train_idx   = np.where(idx != 0)
test_idx    = np.where(idx == 0)

X_train = X[train_idx]
X_test  = X[test_idx]

y_train = synergy_scores[train_idx]
y_test  = synergy_scores[test_idx]


# Save
print("Saving training dataset")
saveFile = open("../datasets/train_data.p", 'wb')
pickle.dump((X_train, y_train), saveFile)
saveFile.close()

print("Saving test dataset")
saveFile = open("../datasets/test_data.p", 'wb')
pickle.dump((X_test, y_test), saveFile)
saveFile.close()
