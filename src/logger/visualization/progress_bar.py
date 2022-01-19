from tqdm import tqdm

class ProgressBar:
    def __init__(self, epoch, max_epochs, num_samples):
        
        self.pbar = tqdm(
            total=num_samples, 
            desc=f'Epoch {epoch}/{max_epochs}' ,unit=' samples')
        
    def update(self, update_steps, metrics=None):
        if metrics is not None:
            postfix = str(metrics).strip('\{\}\'')
            self.pbar.set_postfix_str(postfix, refresh=True)
        
        self.pbar.update(update_steps)

    def close(self):
        self.pbar.close()
