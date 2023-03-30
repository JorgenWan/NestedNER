
from lr_schedulers.base_lr_scheduler import Base_LR_Scheduler
from . import register_lr_scheduler


@register_lr_scheduler("linear_with_warmup")
class Linear_With_Warmup(Base_LR_Scheduler):

    def __init__(self, cfg, optimizer):
        super().__init__(optimizer)
        self.lr = cfg.lr
        self.other_lr = cfg.other_lr
        self.warmup_steps = cfg.warmup_steps

        self.max_epoch = cfg.max_epoch
        self.data_size = cfg.data_size
        self.batch_size = cfg.batch_size
        self.update_freq = cfg.update_freq
        self.t_steps = cfg.t_steps

        lr = self.lr * 1 / (self.warmup_steps + 1e-13)
        other_lr = self.other_lr * 1 / (self.warmup_steps + 1e-13)

        assert len(self.optimizer.optimizer.param_groups) == 2
        self.optimizer.optimizer.param_groups[0]["lr"] = lr  # bert
        self.optimizer.optimizer.param_groups[1]["lr"] = other_lr  # other

        # self.optimizer.set_lr(self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--warmup_steps', default=0.01, type=float)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return cls(cfg, optimizer)

    def step_epoch(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate at the end of the given update."""

        progress = num_updates /self.t_steps
        if progress < self.warmup_steps:
            k = progress / self.warmup_steps
        else:
            k = max((progress - 1.) / (self.warmup_steps - 1.), 0.)
        lr = max(0.0, self.lr * k)
        # self.optimizer.set_lr(lr)

        self.optimizer.optimizer.param_groups[0]["lr"] = lr  # bert
        self.optimizer.optimizer.param_groups[1]["lr"] = lr * self.other_lr / self.lr  #
        # other

        return self.optimizer.get_lr()

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {}

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        pass