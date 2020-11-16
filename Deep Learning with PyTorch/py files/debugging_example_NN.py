import torch
import torchvision
import torchvision.transforms as transforms 
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import Dataset

import numpy as np

class Network(nn.Module): # extending nn.Module base class
    def __init__(self):
        super(Network, self).__init__() # initializing base class
        # prebuilt layers
        # 1 input channel, convolved by 6 different filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        
        # fully connected, or dense layers 
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
        
    def forward(self, t):
        # (1) input layer
        t = t
        
        # (2) hidden conv layer
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (3) hidden conv layer
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        
        # (4) hidden layer reshape
        # 4*4 -> height * width -> reduction due to conv operations
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        
        # (5) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)
        
        # (6) output layer (10 classes)
        # using the softmax fucntion which returns a positive probability that sum to 1
        # but we won't use it here because it will be there in the loss function
        # which will implicitly execute the softmax function 
        t = self.out(t)
        # t = F.softmax(t, dim=1) -> done in the loss part implicitly
        
        return t

network = Network()

train_set = torchvision.datasets.FashionMNIST(
    root='.Documents/data/FashionMNIST'
     ,train=True
    ,download=True # downloads it locally (checks existence beforehand)
    ,transform=transforms.Compose([
        transforms.ToTensor() # butilt in tensor transformer
    ])
)

sample = next(iter(train_set))
image, label = sample

output = network(image.unsqueeze(0))
print(output)