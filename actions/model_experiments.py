"""
# Author: ruben 
# Date: 9/2/22
# Project: CACFramework
# File: model_experiments.py

Description: Script to run experiments managed by classes
"""
import logging

import torch

from constants.path_constants import *
from constants.train_constants import IMAGE_TYPE, CUSTOM_NORMALIZED, MODEL_SEED

from utils.io import get_architecture_by_model
from utils.cnn import Architecture, evaluate_model
from utils.fold_handler import FoldHandler
from utils.load_dataset import get_custom_normalization, load_and_transform_data


class Experiment:
    """
    Base class to define common interface to all experiments
    """

    def __init__(self, model_path):
        self._architecture = get_architecture_by_model(model_path)
        self._images = IMAGE_TYPE
        self._path = model_path
        self._init_device()
        self._generate_fold_data()

    def _generate_fold_data(self):
        """
        Creates fold structure where train images will be split in 4 train folds + 1 test fold
        """
        if self._images == 'cropped':
            logging.info(f'Using cropped images')
            self._fold_dataset = ROOT_ORIGINAL_CROP_FOLDS

        else:
            logging.info(f'Using original dimensioned images')
            self._fold_dataset = ROOT_ORIGINAL_FOLDS

        self._fold_handler = FoldHandler(self._fold_dataset, DYNAMIC_RUN_FOLDER)

    def _init_device(self):
        """
        Initialize either cuda or cpu device where train will be run
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {self._device}')

    def _load_model(self, fold_id):
        """
        Loads corresponding fold model
        :param fold_id:
        :return:
        """
        file_path = os.path.join(self._path, f'model_folder_{fold_id}.pt')
        a = Architecture(self._architecture, path=file_path, seed=MODEL_SEED)
        self._model = a.get()

    def _get_normalization(self):
        """
        Retrieves custom normalization if defined.
        :return: ((list) mean, (list) std) Normalized mean and std according to train dataset
        """
        # Generate custom mean and std normalization values from only train dataset
        self._normalization = get_custom_normalization() if CUSTOM_NORMALIZED else (None, None)

    def _iterate_folds(self):
        self._derived_info = dict.fromkeys(list(range(1,6)))
        for fold_id in range(1, 6):
            logging.info(f'Processing fold: {fold_id}')

            # Generates fold datasets
            _, test_data = self._fold_handler.generate_run_set(fold_id)

            # Load fold model
            self._load_model(fold_id)

            # Compute image normalization
            self._get_normalization()

            # Test model. Test step over train model in current fold
            # <--------------------------------------------------------------------->
            test_data_loader = load_and_transform_data(os.path.join(DYNAMIC_RUN_FOLDER, TEST),
                                                       mean=self._normalization[0],
                                                       std=self._normalization[1])
            # Measure test time
            model_performance, fold_accuracy, fold_test_info = evaluate_model(self._model, test_data_loader, self._device, fold_id)
            self._derived_info[fold_id] = fold_test_info
            print(fold_accuracy, fold_test_info)

            self._gather_data()


            break

    def _gather_data(self):
        """
        Override in derived class
        """
        pass

class Experiment1():
    """
    Experiment 1. DL Arch. Comparison: CI(95%)
    ------------------------------------------------------
    - Iterate over folds each train/test steps, each architecture (VGG16, VGG19, ResNet) {**Optional Another architecture**}
        - On ech iteration, gather test_folder accuracy per each architecture (current folds_acc)
    --> Expected output: table with mean, std and confidence interval of each architecture
    """

    def __init__(self, model_path):
        Experiment.__init__(self, model_path)
        self._out_folder = os.path.join(OUTPUT_FOLDER, 'experiment1.latex')

    def _gather_data(self):
        logging.info("Derived gather data")
        print(self._derived_info)

    def _postprocessing(self):
        logging.info("Derived postprocessing")

    def run(self):
        self._iterate_folds()
        self._postprocessing()

class Experiment2(Experiment):
    """
    Experiment 1. DL Arch. Comparison: CI(95%)
    ------------------------------------------------------
    - Iterate over folds each train/test steps, each architecture (VGG16, VGG19, ResNet) {**Optional Another architecture**}
        - On ech iteration, gather test_folder accuracy per each architecture (current folds_acc)
    --> Expected output: table with mean, std and confidence interval of each architecture
    """

    def __init__(self, model_path):
        Experiment.__init__(self, model_path)
        self._out_folder = os.path.join(OUTPUT_FOLDER, 'experiment2.latex')

    def _gather_data(self):
        logging.info("Derived gather data")
        print(self._derived_info)

    def _postprocessing(self):
        logging.info("Derived postprocessing")

    def run(self):
        self._iterate_folds()
        self._postprocessing()


'''


Experiment 2: DL VGG16 (Metrics) 
------------------------------------------------------
- Iterate over folds each test steps:
    - Load trained model
    - For test sample, gather model prediction and build fold vector predictions

- Test against ground truth and get metrics 

--> Expected output: Accuracy, precision, recall, f1 and confusion matrix, independently between eyes
-->  {**Optional tray to get CI o each fold **} 

Experiment 3: Classifiers: (Age, RD, Age + RD) CI(95%)
------------------------------------------------------
- Iterate over the nine classifiers
    - For each one, get accuracy with the different variables
    - Check against article table if we can assert the use of CI. Try to put std dev??

--> Expected output: Accuracy of each classifier with 3 different variables settings: age, RD, Age+RD 

Experiment 4: Conservative protocol (metrics)
------------------------------------------------------
- Assert same patients from clinical and image datasets 
- Iterate over folds each test steps:
    - Load trained model
    - For test sample, gather model prediction and build fold vector predictions

- Gather classifier predictions over folds
- Apply Conservative protocol and get protocol vector
- Test image, clinical and protocol against ground truth and get metrics

Experiment 5: Saving protocol(metrics)
------------------------------------------------------

- Assert same patients from clinical and image datasets 
- Iterate over folds each test steps:
    - Load trained model
    - For test sample, gather model prediction and build fold vector predictions

- Gather classifier predictions over folds
- Apply Saving protocol and get protocol vector
- Test image, clinical and protocol against ground truth and get metrics

'''
