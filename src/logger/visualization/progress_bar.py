from tqdm import tqdm

class ProgressBar:
    def __init__(self, epoch, max_epochs, iterations):
        
        self.pbar = tqdm(
            total=iterations, 
            desc=f'Epoch {epoch}/{max_epochs}:',
            unit=' it',
            bar_format='{desc:12}{percentage:3.0f}%|{bar:12}{r_bar}', 
        )
        
    def update(self, metrics=None):
        if metrics is not None:
            postfix = str(metrics).strip('\{\}\'')
            self.pbar.set_postfix_str(postfix, refresh=True)
        
        self.pbar.update(1)

    def close(self):
        self.pbar.close()
