import pickle
import gzip
import torch
import pandas as pd
import matplotlib.pyplot as plt
from src.model import SynergyNetwork

n_datapoints = 1000


## Load the input data
loadFile = gzip.open("../datasets/X.p.gz")

X = pickle.load(loadFile)
loadFile.close()

X = X[:n_datapoints]
X = torch.from_numpy(X)
X = X.float()

saveFile = open("../datasets/x.p", 'wb')
pickle.dump(X, saveFile)
saveFile.close()


## Load the output data
labels = pd.read_csv("../datasets/labels.csv", index_col=0)

synergy_scores = labels.values[:,3].astype('float')
synergy_scores = synergy_scores[:n_datapoints]
# synergy_scores = synergy_scores.to
synergy_scores = torch.from_numpy(synergy_scores)
synergy_scores = synergy_scores.float()

saveFile = open("../datasets/synergy_scores.p", 'wb')
pickle.dump(synergy_scores, saveFile)
saveFile.close()





