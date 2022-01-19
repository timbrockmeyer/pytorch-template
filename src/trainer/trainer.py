import torch

from .base_trainer import BaseTrainer
from ..utils.metrics import accuracy


class Trainer(BaseTrainer):
    ''' 
    Class for model training, validation and testing. 
    '''
    def __init__(self, args):
        super().__init__(args)

        self.criterion = torch.nn.CrossEntropyLoss()

        # optimizer parameters
        self.lr=args.lr
        self.betas=args.betas
        self.weight_decay=args.weight_decay

    def _get_optimizer(self, model):

        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.lr, 
            betas=self.betas, 
            weight_decay=self.weight_decay
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
        loss = self.criterion(label_predictions, true_labels)

        # calculate metrics
        acc = accuracy(label_predictions, true_labels)

        return {'loss': loss, 'acc': round(acc,4)}