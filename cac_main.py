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
import datetime

from constants.app_constants import *
from constants.path_constants import *
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


def model_train(args, date_time):
    """
    Initializes train stage
    :param args: (list) list of parameters
    :param date_time: (str) date and time to identify execution
    """
    # Print models folder size
    if not args:
        display_help()
        return

    value = request_param('architecture', args)
    architecture = value if value and value in SUPPORTED_ARCHITECTURES else 'vgg16'

    value = request_param('images', args)
    image_type = value if value and value in SUPPORTED_IMAGES else 'cropped'

    logging.info(f'Launching TRAIN step with model: {architecture} over image types: {image_type}')

    mt = ModelTrain(architecture=architecture, images=image_type, date_time=date_time)
    mt.run()


def model_test(args):
    """
    Initializes test stage
    :param args: (list) list of parameters
    """
    if not args:
        display_help()
        return

    value = request_param('images', args)
    image_type = value if value and value in SUPPORTED_IMAGES else 'cropped'

    model = request_param('model', args)
    if os.path.exists(model):
        logging.info(f'Launching TEST step of model stored in: {model} over image types {image_type}')
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
    id_experiment = value if value else '1'

    logging.info(f'Launching Experiment {id_experiment} of model {model} ')


def get_execution_time():
    """
    :return: The current date and time to identify the whole execution
    """
    date_time = datetime.datetime.now()
    return str(date_time.date()) + "_" + str(date_time.time().strftime("%H:%M:%S"))

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
          "      architecture=[vgg16|vgg19|resnet]: Pretrained model to be trained\n" \
          "      images=[original|cropped]: Select input dataset.\n" \
          "  -test | -ts: test trained model.\n" \
          "    [options]:\n" \
          "      model=<model_folder>: model folder root\n" \
          "      dataset=[original|cropped]: Select input dataset.\n" \
          "  -experiments | -ex: Launch specific experiment over a certain train model.\n" \
          "    [options]:\n" \
          "      model=<model_folder>: model folder root\n" \
          "      id=[1|2|3|4|5]: Select experiment id.\n"
    print(msg)


def init_log(date_time, action):
    """
    Inits log with specified datetime and action name
    :param date_time: (str) date time
    :param action: (str) action: train, test, exp
    """
    logging.basicConfig(filename=os.path.join(LOGS_FOLDER, f'{action}_{date_time}.log'), level=logging.INFO,
                        format='[%(levelname)s] : %(message)s')

def cac_main(args):
    """
    Process input arguments and launch corresponding action
    :param args: (list) list of parameters
    """
    date_time = get_execution_time()

    if len(args) < MIN_ARGS or len(args) > MAX_ARGS:
        display_help()
        return

    if args[1] == '-train' or args[1] == '-tr':
        init_log(date_time, 'train')
        model_train(args[2:], date_time)
    elif args[1] == '-test' or args[1] == '-ts':
        init_log(date_time, 'test')
        model_test(args[2:])
    elif args[1] == '-experiments' or args[1] == '-ex':
        init_log(date_time, 'exp')
        model_experiment(args[2:])
    else:
        display_help()
        return


if __name__ == '__main__':
    cac_main(sys.argv)



