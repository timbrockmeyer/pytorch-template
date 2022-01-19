import torch

from time import time
from copy import deepcopy

class BaseTrainer:
    ''' 
    Class for model training, validation and testing. 
    '''
    def __init__(self, args):

        # training parameters
        self.epochs = args.epochs
        self.patience = args.patience

        self.device = args.device
        self.verbose = args.verbose

    def _get_optimizer(self, model):
        # return optimizer object
        raise NotImplementedError

    def _forward_step(self):
        # get training samples and groundtruth labels
        # calculate forward step
        # calculate loss
        # return loss
        raise NotImplementedError

    def _train_iteration(self, model, logger, optimizer, dataloader, lr_scheduler=None):
        '''
        Returns the training loss for each batch of the training dataloader.
        '''
        
        # put model in train mode
        model.train()
        torch.set_grad_enabled(True)

        total_loss = 0
        for i, batch in enumerate(dataloader):
            # call hooks before iteration here
            # ...

            # compute loss and model metrics
            metrics = self._forward_step(model, batch)
            
            # clear gradients
            optimizer.zero_grad()

            # backward
            loss = metrics['loss']
            loss.backward()

            # accumulate loss
            total_loss += loss.item()

            # update parameters
            optimizer.step()

            # learning rate scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

            # logging

            
        total_loss /= i

        return total_loss 
    
    def _validation_iteration(self, model, logger, dataloader):
        '''
        Returns the training loss for each batch of the test dataloader.
        '''
        # put model in test mode
        model.eval()
        torch.set_grad_enabled(False)

        total_loss = 0
        for i, batch in enumerate(dataloader):
            # call hooks before iteration here
            # ...

            # compute metrics
            metrics = self._forward_step(model, batch)
            loss = metrics['loss']

            # accumulate loss
            total_loss += loss.item()

            # logging

            
        total_loss /= i
        
        return total_loss

    def _test_iteration(self, model, logger, dataloader):
        '''
        Returns the training loss for each batch of the test dataloader.
        '''
        # put model in test mode
        model.eval()
        torch.set_grad_enabled(False)

        for i, batch in enumerate(dataloader):
            # call hooks before iteration here
            # ...

            # compute metrics
            metrics = self._forward_step(model, batch)

            # logging


    def fit(self, model, logger, training_dataloader, validation_dataloader=None):
        '''
        Fit the model.
        '''
        optimizer = self._get_optimizer(model)
       
        # early stopping
        early_stopping_counter = 1
        min_loss = float('inf')

        # loop over epochs
        for epoch in range(self.epochs):
            t = time()

            # training step
            loss = self._train_iteration(model, optimizer, logger, training_dataloader)

            # validation step
            if validation_dataloader is not None:
                loss = self._test_iteration(model, logger, validation_dataloader, validation=True)              

            # save best model
            # use best training loss if no validation data is provided 
            if loss < min_loss:
                min_loss = loss
                best_model_state = deepcopy(model.state_dict())
                early_stopping_counter = 1
                improved = True
            else:
                improved = False

            time_elapsed = time() - t
            # console printout
            epoch_printout(epoch, time_elapsed, )
            
                
            # early stopping
            if early_stopping_counter == self.patience:
                # end training
                break
            
            early_stopping_counter += 1
        
        return best_model_state

    def test(self, model, logger, dataloader):
        '''
        Test the model.
        '''
        self._test_iteration(model, logger, dataloader)

    def validate(self, model, logger, dataloader):
        '''
        Validate the model.
        '''
        loss = self._validation_iteration(model, logger, dataloader)

        return loss
        
