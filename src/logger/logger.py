from pathlib import Path
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from visualization import ProgressBar

class Logger:
    def __init__(self, args):
        self.epochs = args.epochs

        # log directory
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = Path(f'runs/{stamp}') 
        tfevent_dir = logdir / 'tensorboard/'
        
        # tensorboard event writer
        self.writer = SummaryWriter(tfevent_dir)      

        # step counters
        self.global_step = 0
        self.epoch_step = 0

        # metric trackers
        self.metric_total = dict()
        self.metric_avg = dict()

    def init_epoch(self, epoch, num_samples):
        # init metric trackers for epoch
        self.metric_total = dict()
        self.metric_avg = dict()
        
        # console output progress
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = ProgressBar(epoch, self.epochs, num_samples)

        # number of batches processed
        self.epoch_step = 0

    def add_graph(self, model, model_input):
        self.writer.add_graph(model, model_input)
        self.writer.close()

    def update_metrics(self, metrics, update_steps):
        for key, value in metrics.items():
            # update accumulator and average
            new_total = self.metric_total.get(key, 0) + value
            new_avg = new_total / self.epoch_step

            # update dicts
            self.metric_total[key] = new_total
            self.metric_avg[key] = new_avg

            # update writer
            self.writer.add_scalar('Loss/train')

        # update progress bar
        self.pbar.update(update_steps, self.metric_avg)

        # increment step counters
        self.global_step += 1
        self.epoch_step += 1

    def close(self):
        self.pbar.close()
