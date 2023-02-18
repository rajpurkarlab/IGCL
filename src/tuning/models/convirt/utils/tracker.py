"""
Tracker utilities to track training progress and handling summary writers.
"""
from collections import defaultdict
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class ProgressTracker():
    def __init__(self, summary_dir, moving_average=100):
        self.losses = []
        self.epoch2losses = defaultdict(list)
        self.loss_history = []
        self.metrics = defaultdict(list)
        self.moving_average = moving_average
        self.summary_dir = summary_dir
        self.writer = SummaryWriter(summary_dir)
    
    def update_loss(self, minibatch_loss, step, epoch=None):
        self.losses.append(minibatch_loss)
        if epoch is not None:
            self.epoch2losses[epoch].append(minibatch_loss)
    
    def write_loss(self, loss_val, step):
        self.writer.add_scalar('Train/loss', loss_val, global_step=step)
    
    def update_metrics(self, metrics, epoch, write_summary=True):
        for metric_name, val in metrics.items():
            self.metrics[metric_name].append(val)
            if write_summary:
                self.writer.add_scalar(f'Valid/{metric_name}', val, global_step=epoch)
    
    def get_avg_loss(self):
        avg_loss = np.mean(self.losses[-self.moving_average:])
        self.loss_history.append(avg_loss)
        return avg_loss
    
    def get_epoch_loss(self, epoch):
        return np.mean(self.epoch2losses[epoch])
    
    def reset_loss(self):
        self.losses = []
        self.step2losses = defaultdict(list)
