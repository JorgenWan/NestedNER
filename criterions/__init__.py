import os
import argparse
import importlib

CRITERION_REGISTRY = {} # criterion_name -> criterion_cls

def register_criterion(criterion_name):

    def register_criterion_cls(criterion_cls):

        CRITERION_REGISTRY[criterion_name] = criterion_cls

        return criterion_cls

    return register_criterion_cls


# when import .models, trigger the decorators and make the registry
# notation:
# 1. __init__.py and __pycache__ should be excluded
#    other *.py files starting with "__" should be avoided
# 2. when register, "criterion_name" should be the same as *.py file

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('__') and not file.startswith('utils'):
        criterion_name = file[:file.find('.py')]
        module = importlib.import_module(f"criterions.{criterion_name}")