import wandb
import numpy as np

class WandbLogger:
    def __init__(self, entity='', project='', name=''):
        wandb.init(
            project=project, 
            entity=entity, 
            name=name
        )

    def plot_metrics(self, metric, value):
        wandb.log({metric: value})
    
    def plot_epoch_loss(self, metric, loss):
        average_loss = np.mean(loss)
        self.plot_metrics(metric, average_loss)