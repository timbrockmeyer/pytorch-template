import sys
import argparse
from distutils.util import strtobool

import torch

from src.dataloader import load_data
from src.model import CNN
from src.trainer import Trainer
from src.logger import Logger

def main(args):

    print()
    # check python and torch versions
    print(f'Python v.{sys.version.split()[0]}')
    print(f'PyTorch v.{torch.__version__}')

    # get device
    device = args.device
    print(f'Device status: {device}')
    
    # data loaders
    loaders = load_data(args)
    train_dataloader, train_size = loaders['train'].values()
    val_dataloader, val_size = loaders['val'].values()
    test_dataloader, test_size = loaders['test'].values()

    sample_inputs, _ = next(iter(train_dataloader))

    # add number of samples to args
    args.train_size = train_size
    args.val_size = val_size
    args.test_size = test_size

    print(f'\nLoading data...')
    print(f'   Training samples: {train_size}')
    if val_dataloader is not None:
        print(f'   Validation samples: {val_size}')
    print(f'   Test samples: {test_size}\n')

    # model
    model = CNN().to(device)

    # torch.autograd.set_detect_anomaly(True) # uncomment for debugging

    trainer = Trainer(args)
    train_logger = Logger(args)
    train_logger.add_graph(model, sample_inputs)

    print('Training...')
    model_state = trainer.fit(model, train_logger, train_dataloader, val_dataloader)
    train_logger.close()

    print('Testing...')
    # load best model state
    model.load_state_dict(model_state)
    trainer.test(model, test_dataloader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ### --- Data params --- ###
    parser.add_argument("-val_split", type=float, default=0.2)
    parser.add_argument("-batch_size", type=int, default=64)

    ### --- Model params --- ###
    # ...
    
    ### --- Training params --- ###
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-patience", type=int, default=10)
    parser.add_argument("-lr", type=float, default=0.0001)
    parser.add_argument("-betas", nargs='*', type=float, default=(0.9, 0.999))
    parser.add_argument("-weight_decay", type=float, default=0)
    parser.add_argument("-device", type=torch.device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # cpu or cuda

    ### --- Logging params --- ###
    parser.add_argument("-log_tensorboard", type=lambda x:strtobool(x), default=False)

    ### --- Other --- ###
    # ...

    args = parser.parse_args()

    main(args)
