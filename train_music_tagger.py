import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pickle
import time

from mscnn import MSCNN
import config

class MusicDataset(Dataset):
	def __init__(self, spectrograms, tags):
		self.spectrograms = spectrograms
		self.tags = tags

	def __len__(self):
		return len(self.tags)

	def __getitem__(self, idx):
		return {'spectrogram': self.spectrograms[idx], 'tags': self.tags[idx]}


def weight_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		nn.init.xavier_normal_(m.weight)
		nn.init.constant_(m.bias, 0)


def evaluate(loader, model, device):
	"""
	Evaluates the model on test/validation data
	:param loader: Data loader
	:param model: The model
	:param device: Device to be used
	:return: The AUC score
	"""
	model.eval()
	score = 0
	with torch.no_grad():
		predictions = []
		targets = []
		for i, data in enumerate(loader, 0):
			batch_inputs = data['spectrogram'].to(device)
			batch_targets = data['tags'].to(device)
			batch_predictions = torch.sigmoid(model(batch_inputs))
			if i == 0:
				predictions = batch_predictions
				targets = batch_targets
			else:
				predictions = np.append(predictions, batch_predictions, axis=0)
				targets = np.append(targets, batch_targets, axis=0)
		score = roc_auc_score(targets, predictions, average="macro")
	return score


def train(train_loader, valid_loader, test_loader, device, weight_init=None):
	"""
	Trains the MSCNN model and saves best performing model on validation set
	:param train_loader: Dataloader for training set
	:param valid_loader: Dataloader for validation set
	:param weight_init: Initialization method for model parameters
	""" 
	print("Using device {} for training".format(device))
	# Initialize the model, training criterion and optimizer
	model = MSCNN(config.N_TAGS).to(device)
	if weight_init is not None:
		model.apply(weight_init) # Initialize parameters with xavier normal
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.L2)

	print("Training Model...")
	best = 0.0 # Best Validation Score
	t_score = 0.0 # Corresponding test score
	cost = 0.0 # Corresponding training cost
	start = time.time()

	for epoch in range(config.EPOCHS):
		model.train()
		start_it = time.time()
		
		# Train
		print("Epoch {}".format(epoch))
		error = []
		for i, data in enumerate(train_loader, 0):
			optimizer.zero_grad()
			inputs = data['spectrogram'].to(device)
			targets = data['tags'].to(device)
			# forward + backward + optimize
			predictions = torch.sigmoid(model(inputs))
			loss = criterion(predictions, targets)
			error.append(loss.data)
			loss.backward()
			optimizer.step()
		print("\tTraining cost is: {}".format(np.mean(error)))
		
		# Evaluate on validation and test set
		valid_score = evaluate(valid_loader, model, device)
		test_score = evaluate(test_loader, model, device)
		print("\tValidation AUC score is: {}".format(valid_score))
		print("\tTest AUC score is: {}".format(test_score))
		if valid_score > best:
			best = valid_score
			t_score = test_score
			cost = np.mean(error)
			torch.save(model.state_dict(), config.DATA_PATH + "trained_model_state_dict.pth") # Save model

		end_it = time.time()
		print("\tTime for {} iteration: {} seconds".format(epoch, end_it - start_it))

	end = time.time()
	print("Best validation score: {}".format(best))
	print("Corresponding training cost: {}".format(cost))
	print("Corresponding test score: {}".format(t_score))
	print("Total time taken is {} seconds".format(end - start))


if __name__ == "__main__":
	# Load data
	tag_freqs, tag_names, vectors = pickle.load(open(config.DATA_PATH + "annotations.pickle", "rb"))
	spectrograms = np.load(open(config.DATA_PATH + "Spectrograms.data", "rb"))

	# Preprocessing and shuffling
	vectors = np.delete(vectors, config.DELETE, axis=0)
	spectrograms = np.delete(spectrograms, config.DELETE, axis=0)
	shuffle_indx = np.arange(vectors.shape[0])
	np.random.shuffle(shuffle_indx)
	vectors = vectors[shuffle_indx][:, :config.N_TAGS].astype(np.float32)
	spectrograms = np.expand_dims(spectrograms[shuffle_indx], axis=1) # For channel

	# Load training data
	train_inps = spectrograms[: config.TRAIN_SIZE]
	train_targets = vectors[: config.TRAIN_SIZE]
	train_set = MusicDataset(train_inps, train_targets)
	train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)


	# Load validation data
	start_pad = config.TRAIN_SIZE
	valid_inps = spectrograms[start_pad: start_pad + config.VALIDATION_SIZE]
	valid_targets = vectors[start_pad: start_pad + config.VALIDATION_SIZE]
	valid_set = MusicDataset(valid_inps, valid_targets)
	valid_loader = DataLoader(valid_set, batch_size=config.VALIDATION_BATCH_SIZE, shuffle=False, num_workers=2)

	# Load test data
	start_pad += config.VALIDATION_SIZE
	test_inps = spectrograms[start_pad: start_pad + config.TEST_SIZE]
	test_targets = vectors[start_pad: start_pad + config.TEST_SIZE]
	test_set = MusicDataset(test_inps, test_targets)
	test_loader = DataLoader(test_set, batch_size=config.TEST_BATCH_SIZE, shuffle=False, num_workers=2)

	# Train model
	device = torch.device("cuda:0" if (config.USE_CUDA and torch.cuda.is_available()) else "cpu")
	train(train_loader, valid_loader, test_loader, device, weight_init=weight_init)