"""
# Author: ruben 
# Date: 26/1/22
# Project: CACFramework
# File: model_train.py

Description: Class to handle train stages
"""
import logging


class ModelTrain():

    def __init__(self, model, dataset):
        """

        :param model: (str) supported model to be trained
        :param dataset: (str) supported dataset as input
        """

        self._model = model
        self._dataset = dataset

        logging.info(f'{self._model} | {self._dataset}')

    def run(self):
        pass
