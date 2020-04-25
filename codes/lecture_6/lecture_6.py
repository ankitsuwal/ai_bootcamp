import torch
import torchvision
import torchvision.transforms as transforms

transforms = transforms.Compose((transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))))

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, transform=transforms, download=True)

testset = torchvision.datasets.CIFAR10(root="./data", train=False, transform=transforms, download=True)

# load data in the code
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
"""
import matplotlib.pyplot as plt 
import numpy as np
def imshow(img):
	img = img / 2 + 0.5
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg)) 
	plt.show()

dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
"""
import torch.nn as nn
import torch.nn.functional as F 

class Net(nn.Module):
	"""docstring for Net"""
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, img):
		x = F.relu(self.pool(self.conv1(img)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16*5*5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()

import torch.optim as optim
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)		


print("starting traing")
for epoch in range(2):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		optimizer.zero_grad()
		output = net(inputs)
		loss = loss_func(output, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		if i%2000 == 1999:
			print(">>: ", i)
			running_loss = 0.0
	print("\nepoch: {0}, loss: {1}".format(epoch+1, running_loss/2000))

print("\nTraining Finished.")