import torch

from tqdm import tqdm
from copy import deepcopy

from ..logger import Logger

class BaseTrainer:
    ''' 
    Class for model training, validation and testing. 
    '''
    def __init__(self):
        pass

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

    def _test_iteration(self, model, dataloader):
        '''
        Returns the training loss for each batch of the test dataloader.
        '''
        # put model in test mode
        model.eval()
        torch.set_grad_enabled(False)
        test_metrics = dict()
        for i, batch in enumerate(tqdm(dataloader, desc='Testing: ', bar_format='{desc:10}{percentage:3.0f}%|{bar:20}{r_bar}')):
            # call hooks before iteration here
            # ...

            # compute metrics
            metrics = self._forward_step(model, batch)
            
            # logging
            metrics.update(loss=metrics['loss'].item())
            
            for key, value in metrics.items():
                # update metrics accumulator 
                test_metrics[key] = test_metrics.get(key, 0) + value
            
        for key, value in test_metrics.items():
            test_metrics[key] = value / (i+1)

        return test_metrics

    def fit(self, model, train_loader, val_loader=[], epochs=50, patience=10, checkpoint=10, logdir='runs/'):
        '''
        Fit the model.
        '''
        optimizer = self._get_optimizer(model)
        
        validate = len(val_loader) > 0 

        iterations = len(train_loader) + len(val_loader) # number of batches per epoch

        # load first batch for tensorboard graph
        sample_inputs, _ = next(iter(train_loader))

        logger = Logger(logdir, epochs, validate=validate)
        logger.add_graph(model, sample_inputs)

        # early stopping
        early_stopping_counter = 1
        min_loss = float('inf')

        # loop over epochs
        for epoch in range(1, epochs + 1):
            ### start of epoch
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
            if early_stopping_counter == patience:
                # end training
                break
            
            early_stopping_counter += 1

            ### end of epoch
            # save checkpoints
            if checkpoint: # if value > 0
                div, mod = divmod(epoch, checkpoint)
                if mod == 0:
                    file_name = f'checkpoint_{div}.pt'
                    logger.save_model(model, file_name)
        
        ### end of training
        logger.save_model(model, 'model.pt')
        logger.close()
        
        return best_model_state

    def test(self, model, dataloader):
        '''
        Test the model.
        '''
        metrics = self._test_iteration(model, dataloader)

        return metrics

    def validate(self, model, dataloader):
        '''
        Validate the model.
        '''
        loss = self._validation_iteration(model, dataloader)

        return loss
        
