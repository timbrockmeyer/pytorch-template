import pathlib

import torch
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor


def load_data(args):

    # dataset arguments
    batch_size = args.batch_size
    val_split = args.val_split
    device = args.device
    data_dir = pathlib.Path(__file__).parent.resolve() / 'data'

    # load data
    train_data = datasets.MNIST(
        root = data_dir,
        train = True,                         
        transform = ToTensor(), 
        download = True,            
    )
    test_data = datasets.MNIST(
        root = data_dir, 
        train = False, 
        transform = ToTensor()
    )

    train_data.data.to(device)
    test_data.data.to(device)
    
    # train/validation split
    train_data_size = len(train_data)
    validation_partition_size = int(val_split * train_data_size)
    train_partition_size = train_data_size - validation_partition_size
    train_subset, validation_subset = torch.utils.data.random_split(train_data, [train_partition_size, validation_partition_size])

    # data loaders
    loaders = {
        'train': DataLoader(
            train_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=1),
        'val': DataLoader(
            validation_subset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=1) if validation_partition_size > 0 else None,
        'test': DataLoader(
            test_data, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=1),
    }

    return loaders