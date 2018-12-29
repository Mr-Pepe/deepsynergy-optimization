import cb18.utils as utils
import cb18.model
import pickle
import torch
import os
import os.path

test_data_path  = "../datasets/test_data.p"
model_paths      = ['/home/felipe/Projects/cb18/saves/search20181224210125/train20181225205131/model807',
                    '/home/felipe/Projects/cb18/saves/search20181224210125/train20181225214100/model518',
                    '/home/felipe/Projects/cb18/saves/search20181224210125/train20181226104223/model697']
model_dir_path  = '/home/felipe/Projects/cb18/saves/gridSearch'

# mode 1: evaluate models on their own
# mode 2: evaluate model mean
mode = 2

if model_paths is None:
    for dirpath, dirnames, filenames in os.walk(model_dir_path):
        for filename in [f for f in filenames if f.find('model') != -1]:
            model_paths.append(os.path.join(dirpath, filename))

# Load test dataset
print("Loading test dataset ... ", end='')
test_data_file = open(test_data_path, 'rb')
X, y = pickle.load(test_data_file)
test_data_file.close()

dataset = utils.Dataset(X, y)
print("Done.")

if mode == 1:

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

elif mode == 2:

    overall_pred_scores = None
    n_overall_pred_scores = 0
    overall_mse = 0
    mse_combined_history = []
    mse_history = []

    for path in model_paths:

        print("Loading model " + path + " ... ", end='')
        model = torch.load(path)
        model.eval()
        print("Done.")

        print("Evaluating model ... ", end='')
        pred_scores = model(X)
        del model
        pred_scores = pred_scores.detach()
        print("Done.")

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

        print("Val score this model: " + "{0:.3f}".format(mse.item()) + "  Combined: " + "{0:.3f}".format(
            mse_combined.item()))