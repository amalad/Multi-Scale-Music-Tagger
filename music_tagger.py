import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from mscnn import MSCNN
import config
import utils as U

model = MSCNN()
inp = torch.rand(2, 1, 128, 628)
out = torch.sigmoid(model(inp))
print(out)

criterion = nn.BCELoss(reduce=False)
optimizer = optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.L2)