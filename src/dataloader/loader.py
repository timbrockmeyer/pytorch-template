import pathlib

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor


def load_data(batch_size, train=True, val_split=0):

    # dataset arguments
    data_dir = pathlib.Path(__file__).parent.resolve() / 'data'

    # load data
    data = datasets.MNIST(
        root = data_dir,
        train = train,                         
        transform = ToTensor(), 
        download = True,            
    )

    # data.data.to(device)
    
    if train:
        # train/validation split
        size = len(data)
        val_size = int(val_split * size)
        train_size = size - val_size
        train_subset, val_subset = torch.utils.data.random_split(data, [train_size, val_size])

        loaders = [DataLoader(data, batch_size, shuffle=True) 
            if len(data) > 0 else list() for data in [train_subset, val_subset]] 
    else:
        # test dataset
        loaders = DataLoader(data, batch_size, shuffle=False)

    return loaders    
