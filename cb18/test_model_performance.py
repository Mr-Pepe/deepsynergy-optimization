import cb18.utils as utils
import cb18.model
import pickle
import torch

test_data_path  = "../datasets/test_data.p"
model_path      = '/home/felipe/Projects/cb18/saves/train20181222002126/model1000'


# Normalize test data according to model means and std_devs
print("Loading test dataset ... ", end='')
test_data_file = open(test_data_path, 'rb')
X, y = pickle.load(test_data_file)
test_data_file.close()

dataset = utils.Dataset(X, y)
print("Done.")

print("Loading model ... ", end='')
model = torch.load(model_path)
model.eval()

print("Evaluating model ... ", end='')
pred_scores = model(X)
print("Done.")

mse = pred_scores[:,0]-y
mse = mse.pow(2)
mse = mse.sum()/len(mse)

print("Test score: " + str(mse.item()))


