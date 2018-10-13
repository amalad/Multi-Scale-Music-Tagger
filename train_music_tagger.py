import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import time

from mscnn import MSCNN
import config

class MusicDataset(Dataset):
	def __init__(self, spectrograms, tags, task="train"):
		self.spectrograms = spectrograms
		self.tags = tags
		self.task = task

	def __len__(self):
		return len(self.tags)

	def __getitem__(self, idx):
		return {'spectrogram': self.spectrograms[idx], 'tags': self.tags[idx]}


def weight_init(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		nn.init.xavier_normal_(m.weight)
		nn.init.xavier_normal_(m.bias)


def train(train_loader, valid_loader, weight_init=None):
	"""
	Trains the MSCNN model and saves best performing model
	:param train_loader: Dataloader for training set
	:param valid_loader: Dataloader for validation set
	:param weight_init: Initialization method for model parameters
	"""
	
	# Initialize the model, training criterion and optimizer
	model = MSCNN(config.N_TAGS)
	if weight_init is not None:
		model.apply(weight_init)
	criterion = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.L2)

	print("Training Model...")
	best = 0.0 # Best Validation Score
	start = time.time()

	for epoch in range(config.EPOCHS):
		model.train()
		start_it = time.time()
		
		# Train
		error = []
		for i, data in enumerate(train_loader, 0):
			optimizer.zero_grad()
			inputs = data['spectrogram']
			targets = data['tags']
			# forward + backward + optimize
			predictions = torch.sigmoid(model(inputs))
			loss = criterion(predictions, targets)
			error.append(loss)
			loss.backward()
			optimizer.step()
		print("Epoch {}".format(epoch))
		print("Training cost is: {}".format(np.mean(error)))
		
		# Evaluate on validation set
		model.eval()
		with torch.no_grad():
			test_predictions = []
			for i, data in enumerate(valid_loader, 0):
				inputs = data['spectrogram']
				targets = data['tags']
				predictions = torch.sigmoid(model(inputs))
				predictions = torch.round(predictions) ##################### Check if this _has_ to be done
				# Append predictions to test_predictions here
				test_predictions.append(___________________)#########################

			# Evaluate with ROC AUC score here
			valid_score = 0 # Edit here ###########################
			print("Validation AUC score is: {}".format(valid_score))
			if valid_score > best:
				best = valid_score
				torch.save(model, config.DATA_PATH + "TrainedModel.pickle")
		end_it = time.time()
		print("Time for {} iteration: {} seconds".format(i, end_it - start_it))

	end = time.time()
	print("Best validation score: {}".format(best))
	print("Total time taken is {} seconds".format(end - start))


if __name__ == "__main__":
	# Load data
	tag_freqs, tag_names, vectors = pickle.load(open(config.DATA_PATH + "annotations.csv", "rb"))
	spectrograms = np.load(open(config.DATA + "Spectrograms.data", "rb"))

	# Preprocessing
	vectors = np.delete(vectors, config.DELETE, axis=0)
	spectrograms = np.delete(vectors, config.DELETE, axis=0)
	shuffle_indx = np.arange(vectors.shape[0])
	np.random.shuffle(shuffle_indx)
	vectors = vectors[shuffle_indx][:, :config.N_TAGS]
	spectrograms = np.expand_dims(spectrograms[shuffle_indx], axis=1) # For channel

	# Load training data
	train_inps = spectrograms[: config.TRAIN_SIZE]
	train_targets = vectors[: config.TRAIN_SIZE]
	train_set = MusicDataset(train_inps, train_targets)
	train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=2)


	# Load validation data
	valid_inps = spectrograms[config.TRAIN_SIZE: config.TRAIN_SIZE + config.VALIDATION_SIZE]
	valid_targets = spectrograms[config.TRAIN_SIZE: config.TRAIN_SIZE + config.VALIDATION_SIZE]
	valid_set = MusicDataset(valid_inps, valid_targets, task="validation")
	valid_loader = DataLoader(valid_set, batch_size=config.VALIDATION_BATCH_SIZE, shuffle=False, num_workers=2)

	# Train model
	train(train_loader, valid_loader, weight_init=weight_init)