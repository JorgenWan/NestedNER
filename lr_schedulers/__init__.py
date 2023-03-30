import os
import argparse
import importlib

LR_SCHEDULER_REGISTRY = {} # lr_scheduler_name -> lr_scheduler_cls

def register_lr_scheduler(lr_scheduler_name):

    def register_lr_scheduler_cls(lr_scheduler_cls):

        LR_SCHEDULER_REGISTRY[lr_scheduler_name] = lr_scheduler_cls

        return lr_scheduler_cls

    return register_lr_scheduler_cls


# when import .models, trigger the decorators and make the registry
# notation:
# 1. __init__.py and __pycache__ should be excluded
#    other *.py files starting with "__" should be avoided
# 2. when register, "lr_scheduler_name" should be the same as *.py file

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('__') and not file.startswith('utils'):
        lr_scheduler_name = file[:file.find('.py')]
        module = importlib.import_module(f"lr_schedulers.{lr_scheduler_name}")