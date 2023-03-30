
class Base_LR_Scheduler:

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        return self.optimizer.get_lr()

    def step_epoch(self, epoch, val_loss=None):
        """Update the learning rate after each epoch."""
        return self.optimizer.get_lr()