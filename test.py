import os
import argparse
from pathlib import Path
from distutils.util import strtobool

import torch

from src.dataloader import load_data
from src.model import CNN
from src.trainer import Trainer

def main(args):

    # torch.autograd.set_detect_anomaly(True) # uncomment for debugging

    # args
    device = args.device
    model_arg = args.model
    
    # model
    model = CNN().to(device)

    # parse model file arg
    rootdir = Path('runs') 
    if model_arg == 'last':
        # sort dirs by date/time
        most_recent_dir = sorted(os.listdir(rootdir), reverse=True)[0]
        model_file = rootdir / most_recent_dir / 'models/' / 'model.pt'
    else:
        # explicit model file string
        model_file = rootdir / model_arg

    # load model state dict
    try:
        model_state = torch.load(model_file)
        model.load_state_dict(model_state)
    except Exception as e:  # TODO: customize exceptions
        if isinstance(e, FileNotFoundError):
            raise FileNotFoundError('Model file not found.')
        elif isinstance(e, ValueError):
            raise ValueError('Model definition and parameters do not match.')     
        else:
            raise Exception('Unknown error while loading state')

    trainer = Trainer(device=args.device)

    # test data
    test_dataloader = load_data(
        batch_size=args.batch_size,
        train=False,
    )

    # load best model state
    results = trainer.test(model, test_dataloader)

    # use or save results here
    print(results)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ### -- Data params --- ###
    parser.add_argument("-batch_size", type=int, default=64)

    ### -- Model params --- ###
    parser.add_argument("-model", type=str, default='last')

    ### --- Logging params --- ###
    parser.add_argument("-log_tensorboard", type=lambda x:strtobool(x), default=False)

    ### --- Other --- ###
    parser.add_argument("-device", type=torch.device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # cpu or cuda

    args = parser.parse_args()
    main(args)
