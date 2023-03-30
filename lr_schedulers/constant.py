
from lr_schedulers.base_lr_scheduler import Base_LR_Scheduler
from . import register_lr_scheduler


@register_lr_scheduler("constant")
class Constant(Base_LR_Scheduler):
    """Decay the LR by a factor every time the validation loss plateaus."""

    def __init__(self, cfg, optimizer):
        super().__init__(optimizer)
        self.lr = cfg.lr
        self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        pass

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return cls(cfg, optimizer)

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        pass