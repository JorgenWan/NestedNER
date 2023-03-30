import os
import argparse
import importlib

MODEL_REGISTRY = {} # model_name -> model_cls
MODEL_WITH_ARCHI = {} # model_name -> list of archi_name

ARCHI_REGISTRY = {} # archi_name -> model_cls
ARCHI_CONFIG_REGISTRY = {} # archi_name -> config_func

def register_model(model_name):

    def register_model_cls(model_cls):

        MODEL_REGISTRY[model_name] = model_cls

        return model_cls

    return register_model_cls

def register_model_architecture(model_name, archi_name):

    def register_model_architecture_func(func):

        MODEL_WITH_ARCHI.setdefault(model_name, []).append(archi_name)

        ARCHI_REGISTRY[archi_name] = MODEL_REGISTRY[model_name]

        ARCHI_CONFIG_REGISTRY[archi_name] = func

        return func

    return register_model_architecture_func

# when import .models, trigger the decorators and make the registry
# notation:
# 1. __init__.py and __pycache__ should be excluded
#    other *.py files starting with "__" should be avoided
# 2. when register, "model_name" should be the same as *.py file

for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('__') and not file.startswith('utils'):
        model_name = file[:file.find('.py')]
        module = importlib.import_module(f"models.{model_name}")
