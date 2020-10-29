import torch
import torchvision

import torchvision.transforms as transforms

"""
You can use the python debugger in order to inspect your code
"""

train_set = torchvision.datasets.FashionMNIST(
    root='./Documents/data'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])

)

image, label = train_set[0]

print(image.shape)