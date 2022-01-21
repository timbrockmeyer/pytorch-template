import os
import sys
import argparse
import torch

from datetime import datetime
from distutils.util import strtobool
from pathlib import Path

from src.dataloader import load_data
from src.model import CNN
from src.trainer import Trainer

def main(args):

    # torch.autograd.set_detect_anomaly(True) # uncomment for debugging

    print()
    # check python and torch versions
    print(f'Python v.{sys.version.split()[0]}')
    print(f'PyTorch v.{torch.__version__}')

    # get device
    device = args.device
    print(f'Device status: {device}')
    
    # model
    model = CNN().to(device)

    if args.from_checkpoint != '':
        try:
            model_state = torch.load(args.from_checkpoint)
            model.load_state_dict(model_state)
        except:
            raise Exception('Model parameters to not match the checkpoint \
                or checkpoint does not exist.')

    # data loaders
    train_dataloader, val_dataloader = load_data(args, train=True)

    # model training/testing and results logger classes
    trainer = Trainer(args)

    # logs
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = Path(f'runs/{stamp}') 

    print('Training...')
    model_state = trainer.fit(model, train_dataloader, val_dataloader, logdir)

    torch.save(model_state, logdir + 'model.pt')


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
    parser.add_argument("-checkpoint", type=int, default=10)
    parser.add_argument("-from_checkpoint", type=str, default='')


    ### --- Logging params --- ###
    parser.add_argument("-log_tensorboard", type=lambda x:strtobool(x), default=False)

    ### --- Other --- ###
    parser.add_argument("-device", type=torch.device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # cpu or cuda

    args = parser.parse_args()
    main(args)
