import torch
import pandas as pd
import traceback

from pathlib import Path
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


class TensorboardWriter:
    def __init__(self, logdir):
        
        self.writer = SummaryWriter(logdir)

    def add_graph(self, model, model_input):

        self.writer.add_graph(model, model_input)
        self.writer.close()

    def add_metrics(self):

        self.writer.add_scalar('Loss/train')


# extraction function
def tb2pandas(path: str) -> pd.DataFrame:
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

# multi-extraction function
def multi_tb2pandas(event_paths: list) -> pd.DataFrame:
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tb2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs
