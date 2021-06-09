import wandb
import numpy as np

class WandbLogger:
    def __init__(self, entity='', project='', name=''):
        wandb.init(
            project=project, 
            entity=entity, 
            name=name
        )

    def plot_metrics(self, log):
        wandb.log(log)
    
    def plot_epoch_loss(self, metric, loss, episode, step):
        average_loss = np.mean(loss)
        log = {
            metric: average_loss,
            'episode': episode,
            'timestep': step
        }
        self.plot_metrics(log)