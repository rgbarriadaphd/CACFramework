"""
# Author: ruben 
# Date: 9/2/22
# Project: CACFramework
# File: model_experiments.py

Description: Script to run experiments managed by classes
"""
import os.path

from utils.io import get_architecture_by_model
from utils.cnn import Architecture
from constants.path_constants import OUTPUT_FOLDER


class Experiment1:
    """
    DL VGG16 (Metrics) eye independent.
    """
    def __init__(self, model_path):
        self._architecture = get_architecture_by_model(model_path)
        self._out_folder = os.path.join(OUTPUT_FOLDER, 'experiment1.latex')
        self._path = model_path


    def run(self):
        print(f'ARCHITECTURE = {self._architecture}')

        for fold_id in range(1,6):
            file_path = os.path.join(self._path, f'model_folder_{fold_id}.pt')
            a = Architecture(self._architecture, path=file_path)
            self._model = a.get()
            print(a.compute_weights_external(self._architecture, self._model))

