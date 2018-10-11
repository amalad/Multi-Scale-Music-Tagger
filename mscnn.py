import torch
import torch.nn as nn
import torch.nn.functional as F

class MSCNN(nn.Module):
	"""
	A multi scale CNN architecture for music auto-tagging
	"""

	def __init__(self, n_tags=50):
		"""
		Initializes network layers
		:param n_tags: Number of output tags
		"""
		super(MSCNN, self).__init__()
		# First channel, operates directly on input
		self.conv11 = nn.Conv2d(1, 50, (3, 7), padding=(3//2, 7//2))
		self.conv12 = nn.Conv2d(50, 100, (3, 5), padding=(3//2, 5//2))
		self.conv13 = nn.Conv2d(100, 70, (3, 3), padding=(3//2,3//2))
		
		# Second channel
		self.conv21 = nn.Conv2d(1, 100, (3, 5), padding=(3//2, 5//2))
		self.conv22 = nn.Conv2d(100, 70, (3, 3), padding=(3//2, 3//2))

		# Third channel
		self.conv31 = nn.Conv2d(1, 70, (3, 3), padding=(3//2, 3//2))

		self.conv4 = nn.Conv2d(210, 70, (3, 3), padding=(3//2, 3//2))
		self.conv5 = nn.Conv2d(70, 70, (3, 3), padding=(3//2, 3//2))
		self.bn_layer = nn.BatchNorm1d(5040, eps=1e-03, momentum=0.99)
		self.dropout = nn.Dropout(p=0.6)

		# Final FC layers
		self.fc1 = nn.Linear(5040, 500)
		self.fc2 = nn.Linear(500, n_tags)

	def forward(self, x):
		x1 = F.max_pool2d(F.relu(self.conv11(x)), (2, 4))
		x1 = F.max_pool2d(F.relu(self.conv12(x1)), (2, 4))
		x1 = F.max_pool2d(F.relu(self.conv13(x1)), (2, 2))

		x2 = F.avg_pool2d(x, (2, 4)) # Subsample 1
		x3 = F.max_pool2d(x2, (2, 4)) # Subsample 2
		x2 = F.max_pool2d(F.relu(self.conv21(x2)), (2, 4))
		x2 = F.max_pool2d(F.relu(self.conv22(x2)), (2, 2))
		x3 = F.max_pool2d(F.relu(self.conv31(x3)), (2, 2))

		x4 = F.relu(self.conv4(torch.cat((x1, x2, x3), dim=1))) # Concat + 2D Conv
		x5 = F.max_pool2d(F.relu(self.conv5(x4)), (2, 2))
		flat = self.dropout(self.bn_layer(x5.view(x.shape[0], -1))) # Flatten + BN + Dropout
		out = self.fc2(F.relu(self.fc1(flat)))

		return out