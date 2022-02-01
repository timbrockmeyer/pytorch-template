import os
import sys
import argparse
from numpy import isin
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

    # check if pre-trained model should be used
    if args.from_checkpoint != '':
        try:
            model_state = torch.load(args.from_checkpoint)
            model.load_state_dict(model_state)
            print('Loading model parameters...')
        except Exception as e:
            if isinstance(e, ValueError):
                raise ValueError('Model definition and parameters do not match.')
            else:
                raise type(e)('Model checkpoint parameters could not be loaded.')
    else:
        # initialize model parameters here if desired
        pass

    # data loaders
    train_dataloader, val_dataloader = load_data(
        batch_size=args.batch_size,
        train=True,
        val_split=args.val_split
    )

    # model training/testing and results logger classes
    trainer = Trainer(
        args.device,
        lr=args.lr,
        betas=args.betas,
        weight_decay=args.weight_decay,
    )

    # logs
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = Path(f'runs/{stamp}') 

    print('Training...')
    model_state = trainer.fit(
        model=model, 
        train_loader=train_dataloader, 
        val_loader=val_dataloader, 
        epochs=args.epochs,
        patience=args.patience,
        checkpoint=args.checkpoint, 
        logdir=logdir,
    )


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
