from . import OPTIMIZER_REGISTRY

def build_optimizer(cfg, params):
    optimizer_name = cfg.optimizer
    optimizer = OPTIMIZER_REGISTRY[optimizer_name].build_optimizer(cfg, params)

    return optimizer
