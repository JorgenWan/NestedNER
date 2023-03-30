import torch.nn as nn

class Base_Criterion(nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    @classmethod
    def build_criterion(cls, cfg):
        return cls(cfg)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss (sum)
        2) the sample size, used as the denominator for the gradient
        3) log outputs for displaying while training
        """
        raise NotImplementedError

    def aggregate_log_outputs(self, log_outputs):
        """Aggregate log outputs from data parallel training."""
        raise NotImplementedError