import cb18.utils as utils
import cb18.model
import pickle

test_data_path  = "../datasets/test_data.p"
model_path      = ''


# Normalize test data according to model means and std_devs
print("Loading test dataset ... ", end='')
test_data_file = open(test_data_path, 'rb')
X, y = pickle.load(test_data_file)
test_data_file.close()

dataset = utils.Dataset(X, y)
print("Done.")

print("Loading model ... ", end='')
model_file = open(model_path, 'rb')
model = pickle.load(model_file)
model_file.close()

means = model.means
std_devs = model.std_devs

print("Normalizing dataset ... ", end='')
X, _, _ = utils.normalize(X, means=means, std_devs=std_devs)
print("Done.")

print("Evaluating model ...")
pred_scores = model(X)
# MSE calculation here
