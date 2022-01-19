import pandas as pd
import traceback

from pathlib import Path
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class Logger:
    def __init__(self, args):
        
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = Path(f'runs/{stamp}') 
        
        self._data = pd.DataFrame()

        self.global_step = 0
        
        # log directory
        dir = logdir / 'tensorboard/'
        self.writer = TensorboardWriter(dir)

    def update(self, metrics_dict):
        
        for key, value in metrics_dict:
            pass

        self.global_step += 1

from tqdm import tqdm
from time import sleep

pbar = tqdm(total=10 ,unit=' samples')
pbar.set_description(f'Epoch {1}/10', refresh=True)

for i in range(10):
    pbar.set_postfix_str(f'Loss: {i+1}/10', refresh=True)
    sleep(0.1)
    pbar.update(1)
pbar.close()


        
