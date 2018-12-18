import pickle
import gzip
import torch
import pandas as pd


## Load the input data
loadFile = gzip.open("../datasets/X.p.gz")

X = pickle.load(loadFile)
loadFile.close()

X = torch.from_numpy(X)
X = X.float()

# Normalize + tanh
means       = X.mean(dim=0)
std_devs    = X.std(dim=0)
mask = std_devs != 0

X = X-means
X[:,mask] = X[:,mask]/std_devs[mask]
X.tanh_()

saveFile = open("../datasets/x_norm_tanh.p", 'wb', pickle.HIGHEST_PROTOCOL)
pickle.dump(X, saveFile)
saveFile.close()


## Load the output data
labels = pd.read_csv("../datasets/labels.csv", index_col=0)

synergy_scores = labels.values[:,3].astype('float')
synergy_scores = torch.from_numpy(synergy_scores)
synergy_scores = synergy_scores.float()
synergy_scores = torch.cat([synergy_scores, synergy_scores], dim=0)

saveFile = open("../datasets/synergy_scores.p", 'wb')
pickle.dump(synergy_scores, saveFile)
saveFile.close()





