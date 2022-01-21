import torch

from copy import deepcopy

from ..logger import Logger

class BaseTrainer:
    ''' 
    Class for model training, validation and testing. 
    '''
    def __init__(self, args):

        # training parameters
        self.epochs = args.epochs
        self.patience = args.patience

        self.device = args.device

    def _get_optimizer(self, model):
        # return optimizer object
        raise NotImplementedError

    def _forward_step(self):
        # get training samples and groundtruth labels
        # calculate forward step
        # calculate loss
        # return loss
        raise NotImplementedError

    def _train_iteration(self, model, optimizer, logger, dataloader, lr_scheduler=None):
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
            metrics.update(loss=loss.item())
            logger.update(metrics)

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
            metrics.update(loss=loss.item())
            logger.update(metrics)

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
            metrics.update(loss=metrics['loss'].item())
            logger.update(metrics)

    def fit(self, model, train_loader, val_loader=list(), logdir='runs/'):
        '''
        Fit the model.
        '''
        optimizer = self._get_optimizer(model)
        
        validate = len(val_loader) > 0
        # number of batches per epoch
        iterations = len(train_loader) + len(val_loader) 

        # load first batch for tensorboard graph
        sample_inputs, _ = next(iter(train_loader))

        logger = Logger(logdir, self.epochs, validate=validate)
        logger.add_graph(model, sample_inputs)

        # early stopping
        early_stopping_counter = 1
        min_loss = float('inf')

        # loop over epochs
        for epoch in range(1, self.epochs + 1):
            # initialize logger
            logger.train()
            logger.new_epoch(epoch, iterations)

            # training step
            loss = self._train_iteration(model, optimizer, logger, train_loader)

            # validation step
            if validate:
                logger.eval()
                loss = self._validation_iteration(model, logger, val_loader)              

            # save best model
            # use best training loss if no validation data is provided 
            if loss < min_loss:
                min_loss = loss
                best_model_state = deepcopy(model.state_dict())
                early_stopping_counter = 1         
                
            # early stopping
            if early_stopping_counter == self.patience:
                # end training
                break
            
            early_stopping_counter += 1
        
        logger.close()
        
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
        
