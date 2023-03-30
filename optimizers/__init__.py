import os
import argparse
import importlib

OPTIMIZER_REGISTRY = {} # optimizer_name -> optimizer_cls

def register_optimizer(optimizer_name):

    def register_optimizer_cls(optimizer_cls):

        OPTIMIZER_REGISTRY[optimizer_name] = optimizer_cls

        return optimizer_cls

    return register_optimizer_cls


# when import .models, trigger the decorators and make the registry
# notation:
# 1. __init__.py and __pycache__ should be excluded
#    other *.py files starting with "__" should be avoided
# 2. when register, "optimizer_name" should be the same as *.py file

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('__') and not file.startswith('utils'):
        optimizer_name = file[:file.find('.py')]
        module = importlib.import_module(f"optimizers.{optimizer_name}")