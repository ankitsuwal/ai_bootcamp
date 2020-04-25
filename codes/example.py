import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

# you can download iris dataset from 
# wget https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv
# read csv file and prepare matrix like:  
"""[epal_length,
    sepal_width,
    petal_length]""" # and predict petal_width

# [row1,
# row2,
# row3]

class IrisNet(nn.Module):
    """docstring for IrisNet"""
    def __init__(self):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(3,4)
        self.fc2 = nn.Linear(4,3)
        self.fc3 = nn.Linear(3,1)
    
    def forward(x):
        # Neural net takes an input and predict a value
        # using relu as activation function
        x = F.relu(self.fc1(x))         
        x = F.relu(self.fc2(x))         
        x = F.relu(self.fc3(x))
        return x        

# loss function: hels us in deciding how far are we from ground truth.
# optimizer: help us in backprop. is a way to tune the weight in the NN.

neural_net = IrisNet()
loss = nn.MSELoss()
optimizer = optim.sGD(neural_net.parameters(), lr=0.01)

epoch = 5
for i in range(epoch):
    input_arr, ground_truth_value = data
    optimizer.zero_grad()
    #forward_prop
    output = neural_net(input_arr)
    loss_calc = loss(output, ground_truth_value)
    #backward prop
    loss_calc.backward()  #gradient calculation 
    optimizer.step()  #take step 