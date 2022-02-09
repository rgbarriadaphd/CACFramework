"""
# Author: ruben 
# Date: 9/2/22
# Project: CACFramework
# File: io.py

Description: Functions to handle I/O operations
"""


def get_architecture_by_model(model):
    """
    Returns model architecture
    :param model: (str) Model folder path
    :return: Architecture name by analyzing path name
    """
    return model.split('_')[1]

