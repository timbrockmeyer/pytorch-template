import sys
import argparse
from distutils.util import strtobool

import torch

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

    trainer = Trainer(args)

    print('Testing...')
    # test data
    test_dataloader = load_data(args, train=False)
    
    # load model state
    model.load_state_dict(model_state)
    trainer.test(model, test_dataloader)

    print('Testing...')
    # load best model state
    model.load_state_dict(model_state)
    trainer.test(model, test_dataloader)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ### -- Data params --- ###
    parser.add_argument("-batch_size", type=int, default=64)


    ### --- Logging params --- ###
    parser.add_argument("-log_tensorboard", type=lambda x:strtobool(x), default=False)

    ### --- Other --- ###
    parser.add_argument("-device", type=torch.device, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # cpu or cuda

    args = parser.parse_args()
    main(args)
