import torch

from .base_trainer import BaseTrainer
from ..utils.metrics import accuracy


class Trainer(BaseTrainer):
    ''' 
    Class for model training, validation and testing. 
    '''
    def __init__(self, device, **kwargs):
        super().__init__()

        self.device = device

        self._lr = kwargs.get('lr', None)
        self._betas = kwargs.get('betas', None)
        self._weight_decay = kwargs.get('weight_decay', None)
        
        self._criterion = torch.nn.CrossEntropyLoss()

    def _get_optimizer(self, model):

        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self._lr, 
            betas=self._betas, 
            weight_decay=self._weight_decay
        )
        return optimizer

    def _forward_step(self, model, batch):
        
        # unpack training samples and groundtruth labels
        samples, true_labels = batch

        samples.data.to(self.device)
        true_labels.data.to(self.device)

        # forward step
        label_predictions = model(samples)
        
        # calculate loss
        loss = self._criterion(label_predictions, true_labels)

        # calculate metrics
        acc = accuracy(label_predictions, true_labels)

        return {'loss': loss, 'acc': round(acc,4)}