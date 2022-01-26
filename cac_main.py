"""
# Author: ruben 
# Date: 26/1/22
# Project: CACFramework
# File: cac_main.py

Description: Main Script to centralize main runs (train/test) and experiments
"""
import os.path
import sys
import logging

from constants.app_constants import *
from actions.model_train import ModelTrain


def request_param(param, args):
    """
    Provides requested parameter if exists in the argument list
    :param param: (str) Requested parameter
    :param args: (list) list of parameters
    :return: parameter value, None otherwise
    """
    for arg in args:
        if param in arg:
            return arg.split('=')[1]
    return None


def model_train(args):
    """
    Initializes train stage
    :param args: (list) list of parameters
    """
    # Print models folder size
    if not args:
        display_help()
        return

    value = request_param('model', args)
    model = value if value and value in SUPPORTED_MODELS else 'vgg16'

    value = request_param('dataset', args)
    dataset = value if value and value in SUPPORTED_DATASET else 'cropped'

    logging.info(f'Launching TRAIN step with model: {model} over dataset: {dataset}')

    mt = ModelTrain(model=model, dataset=dataset)
    mt.run()


def model_test(args):
    """
    Initializes test stage
    :param args: (list) list of parameters
    """
    if not args:
        display_help()
        return

    value = request_param('dataset', args)
    dataset = value if value and value in SUPPORTED_DATASET else 'cropped'

    model = request_param('model', args)
    if os.path.exists(model):
        logging.info(f'Launching TEST step of model stored in: {model} over dataset {dataset}')
    else:
        logging.info(f'Model: {model} cannot be found')


def model_experiment(args):
    """
    Initializes experiments stage
    :param args: (list) list of parameters
    """
    if not args:
        display_help()
        return

    model = request_param('model', args)
    if os.path.exists(model):
        logging.info(f'Launching TEST step of model stored in: {model}')
    else:
        logging.info(f'Model: {model} cannot be found')

    value = request_param('id', args)
    id = value if value else '1'

    logging.info(f'Launching Experiment {id} of model {model} ')


def display_help():
    """
    Prints usage information
    """
    msg = "CAC Framework\n" \
          "==========================\n" \
          "usage: python cac_main.py action [options]\n" \
          "[actions]:\n" \
          "  -help | -h: Display help.\n" \
          "  -train | -tr: train defined model with input hyperparams.\n" \
          "    [options]:\n" \
          "      model=[vgg16|vgg19|resnet]: Pretrained model be trained\n" \
          "      dataset=[original|cropped]: Select input dataset.\n" \
          "  -test | -ts: test trained model.\n" \
          "    [options]:\n" \
          "      model=<model_folder>: model folder root\n" \
          "      dataset=[original|cropped]: Select input dataset.\n" \
          "  -experiments | -ex: Launch specific experiment over a certain train model.\n" \
          "    [options]:\n" \
          "      model=<model_folder>: model folder root\n" \
          "      id=[1|2|3|4|5]: Select experiment id.\n"
    print(msg)


def cac_main(args):
    """
    Process input arguments and launch corresponding action
    :param args: (list) list of parameters
    """
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] : %(message)s')

    logging.info("Estamos dentro")
    if len(args) < MIN_ARGS or len(args) > MAX_ARGS:
        display_help()
        return

    if args[1] == '-train' or args[1] == '-tr':
        model_train(args[2:])
    elif args[1] == '-test' or args[1] == '-ts':
        model_test(args[2:])
    elif args[1] == '-experiments' or args[1] == '-ex':
        model_experiment(args[2:])
    else:
        display_help()
        return


if __name__ == '__main__':
    cac_main(sys.argv)



