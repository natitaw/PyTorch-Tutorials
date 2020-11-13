import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from torch.utils.data import DataLoader

mean = 0.28590065240859985
std = 0.3513832092285156

train_set = torchvision.datasets.FashionMNIST(
    root='./Documents/data'
     ,train=True
    ,download=True # downloads it locally (checks existence beforehand)
    ,transform=transforms.Compose([
        transforms.ToTensor(), # butilt in tensor transformer
        # TODO: Normalize
        transforms.Normalize(mean, std)
    ])
)

loader = DataLoader(train_set, batch_size=1)
image, label = next(iter(loader))

print(image.shape)