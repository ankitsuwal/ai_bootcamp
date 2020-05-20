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
from torch.utils.data import DataLoader, Dataset, random_split
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
logger = logging.getLogger()
# determine the supported device
def get_device():
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# TODO: write Dataloader to load
# will load data for disk and it will pass it to GPU for traning 


# device = torch.device("cuda")
# b = torch.tensor([1,2,3])
# print(b.device) # output: cpu

class IrisNet(nn.Module):
    """docstring for Net"""
    def __init__(self, inputs, hidden, output):
        # self.inp = input
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(inputs, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        output = self.fc2(x)
        return output


class IrisDataset(Dataset):
    def __init__(self, data, features, label):
        self.X = data[features].astype(np.float32).values
        self.y = data[label].astype(np.float32).values.reshape(-1, 1)
        # print(self.X, ">>>>", self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return [self.X[item], self.y[item]]


class DataLoad():
    """docstring for DataLoad"""
    def __init__(self, file_path):
        super(DataLoad, self).__init__()
        self.file_path = file_path
        
    def data_loder(self):
        # loading data from csv file using pandas
        print("Loading data from csv")
        names = ['sepal_length', 'sepal_width', 'petal_length']
        to_predict = ['petal_width']
        data = pd.read_csv(self.file_path).iloc[:, 0:4]
        train_data, test_data = train_test_split(data, train_size=0.7)
        print("train_data length: {} and test_data length: {}".format(len(train_data), len(test_data)))

        train_dataset = IrisDataset(train_data, names, to_predict)
        test_dataset = IrisDataset(test_data, names, to_predict)

        return train_dataset, test_dataset


class TrainTest():
    """docstring for TrainTest"""
    def __init__(self, net, train_dataset, test_dataset):
        super(TrainTest, self).__init__()
        self.net = net
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epoch = 2
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()
        self.trainloader = DataLoader(self.train_dataset, batch_size=10, shuffle=True)        
        self.testloader = DataLoader(self.test_dataset, batch_size=1, shuffle=True)        
    
    def train(self):

        for i in range(self.epoch):
            running_loss = 0
            print("11111111111111111111111111")
            for j, data in enumerate(self.trainloader, 0):
                inputs, label = data[0], data[1]
                self.optimizer.zero_grad()
                output = self.net(inputs)
                loss = self.loss_func(output, label)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            logger.info("Epoch: {}, Loss: {}".format(i + 1, running_loss / len(self.trainloader)))
        logger.info("Training complete")

    def test(self):
        running_loss = 0
        with torch.no_grad():
            print("2222222222222222222222222222222")
            for data in self.testloader:
                inputs, label = data[0], data[1]
                output = self.net(inputs)
                loss = self.loss_func(output, label)
                running_loss += loss.item()

            logger.info("Loss on test data: {}".format(running_loss/len(self.testloader)))


if __name__ == "__main__":
    device = get_device()
    print(">>>: {}".format(device))
    # logger.info("Device using: {}".format(device))
    net = IrisNet(3, 10, 1).to(device)
    # print("Created Neural Network: {}".format(net))
    data_path = "/home/dell/work/ai_bootcamp/codes/lecture_7/iris.csv"
    irisd_obj = DataLoad(data_path)
    train_dataset, test_dataset = irisd_obj.data_loder()
    obj = TrainTest(net, train_dataset, test_dataset)
    obj.train()
    obj.test()


