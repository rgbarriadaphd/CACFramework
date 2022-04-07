"""
# Author: ruben
# Date: 26/1/22
# Project: CACFramework
# File: model_train.py

Description: Class to handle train stages
"""
import logging
import torch
import time
from copy import copy, deepcopy
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from string import Template

from constants.path_constants import *
from constants.train_constants import *

from utils.fold_handler import FoldHandler
from utils.cnn import Architecture, train_model, evaluate_model
from utils.load_dataset import get_custom_normalization, load_and_transform_data
from utils.metrics import CrossValidationMeasures


class ModelTrain:

    def __init__(self, architecture, images, date_time):
        """
        Model train constructor initializes al class attributes.
        :param architecture: (str) supported architecture to be trained
        :param images: (str) supported dataset as input
        :param date_time: (str) date and time to identify execution
        """
        self._images = images
        self._date_time = date_time
        self._create_train_folder()
        self._init_device()
        self._generate_fold_data()



    def _create_train_folder(self):
        """
        Creates the folder where output data will be stored
        """
        self._train_folder = os.path.join(MODELS_FOLDER, f'train_{ARCHITECTURE}_{self._date_time}')
        try:
            os.mkdir(self._train_folder)
        except OSError:
            logging.error("Creation of model directory %s failed" % self._train_folder)
        else:
            logging.info("Successfully created model directory %s " % self._train_folder)

    def _init_device(self):
        """
        Initialize either cuda or cpu device where train will be run
        """
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {self._device}')

    def _generate_fold_data(self):
        """
        Creates fold structure where train images will be split in 4 train folds + 1 test fold
        """
        if self._images == 'cropped':
            logging.info(f'Using cropped images')
            self._fold_dataset = ROOT_ORIGINAL_CROP_FOLDS
        elif self._images == 'cropped_nerve':
            logging.info(f'Using nerve cropped images')
            self._fold_dataset = ROOT_ORIGINAL_CROP_FOLDS_NERVE
        else:
            logging.info(f'Using original dimensioned images')
            self._fold_dataset = ROOT_ORIGINAL_FOLDS

        self._fold_handler = FoldHandler(self._fold_dataset, DYNAMIC_RUN_FOLDER)

    def _init_model(self):
        """
        Gathers model architecture
        :return:
        """
        self._architecture = Architecture(ARCHITECTURE, seed=MODEL_SEED)
        self._model = self._architecture.get()

    def _get_normalization(self):
        """
        Retrieves custom normalization if defined.
        :return: ((list) mean, (list) std) Normalized mean and std according to train dataset
        """
        # Generate custom mean and std normalization values from only train dataset
        self._normalization = get_custom_normalization() if CUSTOM_NORMALIZED else (None, None)

    def _save_model_fold(self, fold_model, fold_id):
        """
        Save train model by fold
        :param fold_model: (torch) train model
        :param fold_id: (int) fold id
        """
        folder_model_path = os.path.join(self._train_folder, f'model_folder_{fold_id}.pt')
        logging.info(f'Saving folder model to {folder_model_path}')
        torch.save(fold_model.state_dict(), folder_model_path)

    def _save_plot_fold(self, measures, fold_id, plot_type='loss'):
        """
        Plots fold loss evolution
        :param measures: (list) list of losses
        :param fold_id: (int) fold id
        :param plot_type: (str) type of measure plot
        """
        fig, ax = plt.subplots()

        tx = f'Model = {ARCHITECTURE}\nEpochs = {EPOCHS}\nBatch size = {BATCH_SIZE}\nLearning rate = {LEARNING_RATE}'

        ax.plot(list(range(1, len(measures) + 1)), measures)

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.72, 0.95, tx, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.ylabel(plot_type)
        plt.xlabel('epochs')
        plt.title(f'{plot_type} evolution')
        plot_path = os.path.join(self._train_folder, f'{plot_type}_{fold_id}.png')
        logging.info(f'Saving plot to {plot_path}')
        plt.savefig(plot_path)

    def _save_fold_accuracies(self, train, test, fold_id):
        """
        Plot together train and test accuracies by fold
        :param train: (list) list of train accuracies
        :param test: (list) list of test accuracies
        :param fold_id: (int) fold id
        """
        assert len(train) == len(test)
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('accuracy')
        plt.title(f'Model accuracy')
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

        x_epochs = list(range(1, len(test) + 1))
        ax1.plot(x_epochs, test, label='test')
        ax1.plot(x_epochs, train, label='train')
        ax1.legend()

        plot_path = os.path.join(self._train_folder, f'accuracy_{fold_id}.png')
        logging.info(f'Saving accuracy plot to {plot_path}')
        plt.savefig(plot_path)

    def _save_train_summary(self, folds_performance, global_performance):

        # Global Configuration
        summary_template_values = {
            'datetime': datetime.now(),
            'model': ARCHITECTURE,
            'image_type': IMAGE_TYPE,
            'normalized': CUSTOM_NORMALIZED,
            'save_model': SAVE_MODEL,
            'plot_loss': SAVE_LOSS_PLOT,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'weight_decay': WEIGHT_DECAY,
            'criterion': CRITERION,
            'optimizer': OPTIMIZER,
            'device': self._device,
            'require_grad': REQUIRES_GRAD,
            'weight_init' : WEIGHT_INIT
        }

        for fold in folds_performance:
            summary_template_values.update(fold)
        # Append fold results
        # Append global performance
        summary_template_values.update(global_performance)

        if FOLDS == 'all':
            template = SUMMARY_TEMPLATE_ALL
        elif FOLDS == '1':
            template = SUMMARY_TEMPLATE_FOLD_1
        elif FOLDS == '2':
            template = SUMMARY_TEMPLATE_FOLD_2
        elif FOLDS == '3':
            template = SUMMARY_TEMPLATE_FOLD_3
        elif FOLDS == '4':
            template = SUMMARY_TEMPLATE_FOLD_4
        elif FOLDS == '5':
            template = SUMMARY_TEMPLATE_FOLD_5
        # TODO: complete other folds

        # Substitute values
        with open(template, 'r') as f:
            src = Template(f.read())
            report = src.substitute(summary_template_values)
            logging.info(report)

        # Write report
        with open(os.path.join(self._train_folder, 'summary.out'), 'w') as f:
            f.write(report)

    def run(self):
        """
        Runs train stage
        """
        t0 = time.time()
        folds_performance = []
        folds_acc = []

        # base_weights = self._architecture.weights_sum()
        # logging.info(f'Base model weights: {base_weights}')
        folds_ids = range(1, 6) if FOLDS == 'all' else [int(FOLDS)]

        for fold_id in folds_ids:
            logging.info(f'Processing fold: {fold_id}')

            # Generates fold datasets
            train_data, test_data = self._fold_handler.generate_run_set(fold_id)

            self._init_model()

            # At each iteration the model should remain the same, so conditions are equal in each fold.
            fold_architecture = self._architecture
            fold_model = fold_architecture.get()
            fold_model.to(device=self._device)
            fold_weights = fold_architecture.weights_sum()
            fold_weights = fold_architecture.compute_weights_external(ARCHITECTURE, fold_model)
            logging.info(f'Pre train step fold model weights: {fold_weights}')
            # assert base_weights == fold_weights

            # Get dataset normalization mean and std
            self._get_normalization()

            # Train model. Train step over current fold configuration
            # <--------------------------------------------------------------------->
            train_data_loader = load_and_transform_data(os.path.join(DYNAMIC_RUN_FOLDER, TRAIN),
                                                        batch_size=BATCH_SIZE,
                                                        data_augmentation=False,
                                                        mean=self._normalization[0],
                                                        std=self._normalization[1])
            # Measure train time
            t0_fold_train = time.time()
            fold_model, losses, accuracies = train_model(model=fold_model,
                                                         device=self._device,
                                                         train_loader=train_data_loader,
                                                         normalization=self._normalization
                                                         )
            tf_fold_train = time.time() - t0_fold_train

            post_weights = fold_architecture.compute_weights_external(ARCHITECTURE, fold_model)

            logging.info(f'Post train step fold model weights: {post_weights}')

            # Test model. Test step over train model in current fold
            # <--------------------------------------------------------------------->
            test_data_loader = load_and_transform_data(os.path.join(DYNAMIC_RUN_FOLDER, TEST),
                                                       mean=self._normalization[0],
                                                       std=self._normalization[1])
            # Measure test time
            t0_fold_test = time.time()
            model_performance, fold_accuracy, _ = evaluate_model(fold_model, test_data_loader, self._device, fold_id)
            tf_fold_test = time.time() - t0_fold_test
            folds_acc.append(fold_accuracy)

            logging.info(f'Fold {fold_id} accuracy over test set: {fold_accuracy}')

            # Update fold data
            fold_data = {
                f'fold_id_{fold_id}': fold_id,
                f'n_train_{fold_id}': len(train_data_loader.dataset),
                f'n_test_{fold_id}': len(test_data_loader.dataset),
                f'mean_{fold_id}': f'[{self._normalization[0][0]:.{ND}f}, {self._normalization[0][1]:.{ND}f}, {self._normalization[0][2]:.{ND}f}]',
                f'std_{fold_id}': f'[{self._normalization[1][0]:.{ND}f}, {self._normalization[1][1]:.{ND}f}, {self._normalization[1][2]:.{ND}f}]',
                f'fold_train_time_{fold_id}': f'{tf_fold_train:.{ND}f}',
                f'fold_test_time_{fold_id}': f'{tf_fold_test:.{ND}f}',
            }
            model_performance.update(fold_data)
            folds_performance.append(model_performance)

            # Generate Loss plot
            if SAVE_LOSS_PLOT:
                self._save_plot_fold(losses, fold_id, plot_type='loss')

            # Generate Loss plot
            if SAVE_ACCURACY_PLOT:
                self._save_fold_accuracies(accuracies[0], accuracies[1], fold_id)

            # Save fold model
            if SAVE_MODEL:
                self._save_model_fold(fold_model, fold_id)

            # Run only one fold
            if MONO_FOLD:
                logging.info("Only one fold is executed")
                break

        # Compute global performance info
        cvm = CrossValidationMeasures(measures_list=folds_acc, percent=True, formatted=True)
        f_acc = '['
        for p, fa in enumerate(folds_acc):
            f_acc += f'{fa:.{ND}f}'
            if (p + 1) != len(folds_acc):
                f_acc += ','
        f_acc += ']'
        global_performance = {
            'execution_time': str(timedelta(seconds=time.time() - t0)),
            'folds_accuracy': f_acc,
            'cross_v_mean': cvm.mean(),
            'cross_v_stddev': cvm.stddev(),
            'cross_v_interval': cvm.interval()
        }
        self._save_train_summary(folds_performance, global_performance)
