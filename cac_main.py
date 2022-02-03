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
import shutil

from constants.path_constants import *
from constants.train_constants import *
from actions.model_train import ModelTrain


def request_param(param, args):
    """
    Provides requested parameter if exists in the argument list
    :param param: (str) Requested parameter
    :param args: (list) list of parameters
    :return: parameter value, None otherwise
    """
    for arg in args:
        arg = arg[1:] if arg.startswith('-') else arg
        if param in arg:
            return arg.split('=')[1]
    return None


def model_train(date_time):
    """
    Initializes train stage
    :param date_time: (str) date and time to identify execution
    """
    mt = ModelTrain(architecture=ARCHITECTURE, images=IMAGE_TYPE, date_time=date_time)
    mt.run()


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
        return

    value = request_param('id', args)
    id_experiment = value if value else '1'

    logging.info(f'Launching Experiment {id_experiment} of model {model} ')


def get_execution_time():
    """
    :return: The current date and time to identify the whole execution
    """
    date_time = datetime.datetime.now()
    return str(date_time.date().strftime('%Y%m%d')) + "_" + str(date_time.time().strftime("%H%M%S"))


def display_help():
    """
    Prints usage information
    """
    msg = "CAC Framework\n" \
          "==========================\n" \
          "usage: python cac_main.py <options>\n" \
          "Options:\n" \
          "  [-help | -h]: Display help.\n" \
          "  [-clear | -cl]: Clear logs and models.\n" \
          "  [-train | -tr]: train and test model.\n" \
          "  [-experiment | -exp] -id=<exp_id> -model=<model_folder>:  test specified model.\n"
    print(msg)


def init_log(date_time, action):
    """
    Inits log with specified datetime and action name
    :param date_time: (str) date time
    :param action: (str) action: train, test, exp
    """
    logging.basicConfig(filename=os.path.join(LOGS_FOLDER, f'{action}_{date_time}.log'), level=logging.INFO,
                        format='[%(levelname)s] : %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def cac_main(args):
    """
    Process input arguments and launch corresponding action
    :param args: (list) list of parameters
    """
    if len(args) != 2 and len(args) != 4:
        display_help()
        return

    if len(args) == 2 and args[1] == '-clear' or args[1] == '-cl':
        answer_valid = False
        while not answer_valid:
            value = input("Do you want to clear logs and models folders? (y/n):  ")
            if value == 'y' or value == 'yes':
                print("Deleting logs and model storage")
                for f in os.listdir(MODELS_FOLDER):
                    shutil.rmtree(os.path.join(MODELS_FOLDER, f))
                for f in os.listdir(LOGS_FOLDER):
                    os.remove(os.path.join(LOGS_FOLDER, f))
                answer_valid = True
            elif value == 'n' or value == 'no':
                answer_valid = True
            else:
                print(f'Please, "{value}" is not a valid answer. Please, type "y" or "n" ')
        return

    date_time = get_execution_time()

    if args[1] == '-train' or args[1] == '-tr':
        init_log(date_time, 'train')
        model_train(date_time)
    elif args[1] == '-experiments' or args[1] == '-exp':
        init_log(date_time, 'exp')
        model_experiment(args[2:])
    else:
        display_help()
        return


if __name__ == '__main__':
    cac_main(sys.argv)
