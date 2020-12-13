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



"""
- The DataLoader class implements a lot of python functions, such as next, getitem, etc, os that we can use it to manipulate data
- If Shuffle value is True, then the DataLoader will use a random sampler
    - Otherwise a sequencer will be used
- The next() method is implemented in the PyTorch module, it's not the python function
- There is a get_item() method in the data class that defines how the data is indexed
    - The MNIST Class loads all data on memory (which means you need enough memory)
- The alternative is to read from disk using file path data
    - This is implemented in the __getitem__(self, index) method
"""