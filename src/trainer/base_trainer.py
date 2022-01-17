import torch

from copy import deepcopy

class BaseTrainer:
    ''' 
    Class for model training, validation and testing. 
    '''
    def __init__(self, args):

        # training parameters
        self.epochs = args.epochs
        self.patience = args.patience

    def _get_optimizer(self, model):

        # return optimizer object

        raise NotImplementedError

    def _forward_step(self):
        
        # get training samples and groundtruth labels

        # calculate forward step

        # calculate loss

        # return loss

        raise NotImplementedError

    def _train_iteration(self, model, optimizer, dataloader):
        '''
        Returns the training loss for each batch of the training dataloader.
        '''
        
        # put model in train mode
        model.train()
        torch.set_grad_enabled(True)

        losses = []
        for batch in dataloader:
            # call hooks before iteration here
            # ...

            # compute loss
            loss = self._forward_step(model, batch)
            
            # clear gradients
            optimizer.zero_grad()

            # backward
            loss.backward()

            # update parameters
            optimizer.step()

            losses.append(loss.cpu().item())

        return losses
    
    def _test_iteration(self, model, dataloader):
        '''
        Returns the training loss for each batch of the test dataloader.
        '''
        # put model in test mode
        model.eval()
        torch.set_grad_enabled(False)

        losses = []
        for batch in dataloader:
            # call hooks before iteration here
            # ...

            # compute loss
            loss = self._forward_step(model, batch)

            losses.append(loss.cpu().item())

        return losses

    def fit(self, model, training_dataloader, validation_dataloader=None):
        '''
        Fit the model.
        '''

        optimizer = self._get_optimizer(model)

        # losses for all training steps
        training_loss_history = []
        validation_loss_history = []
        
        # early stopping
        early_stopping_counter = 0
        best_loss = float('inf')

        # loop over epochs
        for epoch in range(self.epochs):
            # logger 
            ...
            # training step
            training_losses = self._train_iteration(model, optimizer, training_dataloader)
            training_loss_history.extend(training_losses)

            # validation step
            if validation_dataloader is not None:
                validation_losses = self._test_iteration(model, validation_dataloader)
                validation_loss_history.extend(validation_losses)

                # check early stopping
                avg_loss = sum(validation_losses) / len(validation_losses)
            
            # use training data if no validation data is provided
            else:
                avg_loss = sum(training_losses) / len(training_losses)
            
            # save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = deepcopy(model.state_dict())
                early_stopping_counter = 0
                
            # early stopping
            if early_stopping_counter == self.patience:
                # end training
                break
            else:
                early_stopping_counter += 1

        results = {
            'training_losses': training_loss_history,
            'validation_losses': validation_loss_history,
            'best_loss': best_loss,
            'model_dict': best_model_state
        }

        return results

    def test(self, model, test_dataloader):
        '''
        Test the model.
        '''
        losses = self._test_iteration(model, test_dataloader)
        avg_loss = sum(losses) / len(losses)

        return avg_loss
