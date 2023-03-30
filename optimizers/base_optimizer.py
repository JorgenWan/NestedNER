import torch

class Base_Optimizer:

    def __init__(self, params):
        self.params = list(params)
        # if isinstance(params[0], dict):
        #     self.params = []
        #     for p in params:
        #         self.params.extend(list(p["params"]))
        # else:
        #     self.params = list(params)

    def get_lr(self):
        """Return the current learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = None
        self.optimizer.zero_grad()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm):
        """Clips gradient norm."""
        for group in self.optimizer.param_groups:
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(group['params'], max_norm)

    # def multiply_grads(self, c):
    #     """Multiplies grads by a constant *c*."""
    #     for p in self.params:
    #         if p.grad is not None:
    #             p.grad.data.mul_(c)

    # def clip_grad_norm(self, max_norm):
    #     """Clips gradient norm."""
    #     if max_norm > 0:
    #         return torch.nn.utils.clip_grad_norm_(self.params, max_norm)
    #     else:
    #         return np.sqrt(sum(p.grad.data.norm()**2 for p in self.params if p.grad is not None))

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.optimizer.param_groups:
                group.update(optimizer_overrides)

    def step(self, closure=None):
        """Performs a single optimization step."""
        self.optimizer.step(closure)