import torch
import torch.optim

from optimizers.base_optimizer import Base_Optimizer
from . import register_optimizer

@register_optimizer('adam')
class Adam(Base_Optimizer):
    def __init__(self, cfg, params):
        super().__init__(params)
        self.lr = cfg.lr
        self.betas = eval(cfg.adam_betas)
        self.eps = cfg.adam_eps
        self.weight_decay = cfg.weight_decay

        self.optimizer = torch.optim.Adam(params=params, lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument('--weight_decay', default=1e-8, type=float, help='weight decay')
        parser.add_argument('--adam_betas', default='(0.9, 0.999)', help='betas for Adam optimizer')
        parser.add_argument('--adam_eps', type=float, default=1e-8, help='epsilon for Adam optimizer')

    @classmethod
    def build_optimizer(cls, cfg, params):
        return cls(cfg, params)

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)
