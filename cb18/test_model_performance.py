import cb18.utils as utils
import cb18.model
import pickle
import torch
import os
import os.path

test_data_path  = "../datasets/test_data.p"
model_path      = None
model_dir_path  = '/home/felipe/Projects/cb18/saves/gridSearch'



if model_path is None:
    model_paths = []
    for dirpath, dirnames, filenames in os.walk(model_dir_path):
        for filename in [f for f in filenames if f.find('model') != -1]:
            model_paths.append(os.path.join(dirpath, filename))
else:
    model_paths = [model_path]

# Load test dataset
print("Loading test dataset ... ", end='')
test_data_file = open(test_data_path, 'rb')
X, y = pickle.load(test_data_file)
test_data_file.close()

dataset = utils.Dataset(X, y)
print("Done.")

best_mse = 0

for path in model_paths:

    print("Loading model " + path + " ... ", end='')
    model = torch.load(path)
    model.eval()
    print("Done.")

    print("Evaluating model ... ", end='')
    pred_scores = model(X)
    print("Done.")

    mse = pred_scores[:,0]-y
    mse = mse.pow(2)
    mse = mse.sum()/len(mse)


    print("Test score: " + str(mse.item()))

    if best_mse == 0 or mse < best_mse:
        best_mse = mse
        best_model_path = path
        print("New best MSE.")


print("Best model: " + best_model_path)