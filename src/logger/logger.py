import os
import torch

from torch.utils.tensorboard import SummaryWriter

from .visualization import ProgressBar

class Logger:
    def __init__(self, logdir, epochs, validate=False):

        # log directory
        tfevent_dir = logdir / 'tensorboard/'
        os.makedirs(tfevent_dir, exist_ok=True)

        self.model_dir = logdir / 'models'
        os.makedirs(self.model_dir, exist_ok=True)

        # tensorboard event writers
        self._writers = {
            'train': SummaryWriter(
                log_dir=tfevent_dir,
                flush_secs=10, 
            )
        }

        if validate:
            self._writers['val'] = SummaryWriter(
                log_dir=tfevent_dir,
                flush_secs=10, 
            ) 

        # metric trackers
        self._metric_total = dict()
        self._metric_avg = dict()

        # train or validate mode
        self.mode = None

        # step counters
        self._global_step_counters = {
            'train': 0, 
            'val': 0,
        }
        self._epoch_step_counters = {
            'train': 0, 
            'val': 0,
        }


        # progress bar attr
        self._epochs = epochs
        self._pbar = None

    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'val'

    def new_epoch(self, epoch, num_samples):
        self._current_epoch = epoch

        # init metric trackers for epoch
        self._metric_total = dict()
        self._metric_avg = dict()
        
        # console output progress
        if self._pbar is not None:
            self._pbar.close()
        self._pbar = ProgressBar(epoch, self._epochs, num_samples)

        # number of batches processed
        self._epoch_step_counters['train'] = 0
        self._epoch_step_counters['val'] = 0

    def update(self, metrics):
        # increment step counters
        self._global_step_counters[self.mode] += 1
        self._epoch_step_counters[self.mode] += 1
        
        # iterate through all metrics being tracked
        for key, value in metrics.items():
            # rename dict key with mode as prefix
            key = self.mode + '/' + key

            # update accumulator and average
            new_total = self._metric_total.get(key, 0) + value
            new_avg = round(new_total / self._epoch_step_counters[self.mode], 4)

            # update dicts
            self._metric_total[key] = new_total
            self._metric_avg[key] = new_avg

            # update writer
            self._writers[self.mode].add_scalar(key, value, self._global_step_counters[self.mode])

        # update progress bar
        self._pbar.update(self._metric_avg)

    def add_graph(self, model, model_input):
        self._writers['train'].add_graph(model, model_input)

    def save_model(self, model, file_name='model'):
        model_state = model.state_dict()
        dir = self.model_dir / file_name
        torch.save(model_state, dir)
        
    def close(self):
        for writer in self._writers.values():
            writer.flush()
            writer.close()
        self._pbar.close()
