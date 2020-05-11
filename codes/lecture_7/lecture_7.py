# Take 3 feature -> predict petal width
# can we do it via treditional ML?

# https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv

# EDA - Exploratory Data Analysis! 
# Assignment- Create various graphs using Pandas, Matplotlib, seaborn, some-others library.

import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

# Dataloader
# will load data for disk and it will pass it to GPU for traning 


# device = torch.device("cuda")
# b = torch.tensor([1,2,3])
# print(b.device) # output: cpu

class Net(object):
	"""docstring for Net"""
	def __init__(self, arg):
		super(Net, self).__init__()
		fc1 = nn.Linear(3, 10)
		fc2 = nn.Linear(10, 1)

	def forward(self, x):
		z = F.relu(self.fc1(x))
		output = self.fc2(x)
		return output

net = Net()
criterion = nn.MSELoss()
optim = optim.SGD(net.parameters(), lr=0.01)