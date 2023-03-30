from . import LR_SCHEDULER_REGISTRY

def build_lr_scheduler(cfg, optimizer):
    lr_scheduler_name = cfg.lr_scheduler
    lr_scheduler = LR_SCHEDULER_REGISTRY[lr_scheduler_name].build_lr_scheduler(cfg, optimizer)

    return lr_scheduler
