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
from copy import copy
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt
from string import Template

from constants.path_constants import *
from constants.train_constants import *

from utils.fold_handler import FoldHandler
from utils.cnn import Architecture, train_model, eva
from utils.load_dataset import get_custom_normalization, load_and_transform_data


class ModelTrain:

    def __init__(self, architecture, images, date_time):
        """
        Model train constructor initializes al class attributes.
        :param architecture: (str) supported architecture to be trained
        :param images: (str) supported dataset as input
        :param date_time: (str) date and time to identify execution
        """
        self._architecture = architecture
        self._images = images
        self._date_time = date_time
        self._create_train_folder()
        self._init_device()
        self._generate_fold_data()
        self._init_model()

    def _create_train_folder(self):
        """
        Creates the folder where output data will be stored
        """
        self._train_folder = os.path.join(MODELS_FOLDER, "run_" + self._date_time)
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

        else:
            logging.info(f'Using original dimensioned images')
            self._fold_dataset = ROOT_ORIGINAL_FOLDS

        self._fold_handler = FoldHandler(self._fold_dataset, DYNAMIC_RUN_FOLDER)

    def _init_model(self):
        """
        Gathers model architecture
        :return:
        """
        architecture = Architecture(self._architecture, seed=MODEL_SEED)
        self._model = architecture.get()

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

    def _save_plot_fold(self, losses, fold_id):
        """
        Plots fold loss evolution
        :param losses: (list) list of losses
        :param fold_id: (int) fold id
        """

        fig, ax = plt.subplots()
        model = 'vgg16'

        tx = f'Model = {self._architecture}\nEpochs = {EPOCHS}\nBatch size = {BATCH_SIZE}\nLearning rate = {LEARNING_RATE}'

        ax.plot(list(range(len(losses))), losses)

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        ax.text(0.72, 0.95, tx, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=props)

        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.title('Loss convergence')
        plot_path = os.path.join(self._train_folder, f'loss_{fold_id}.png')
        logging.info(f'Saving plot to {plot_path}')
        plt.savefig(plot_path)
    
    def _save_train_summary(self, folds_summary):

        # Global Configuration
        summary_template_values = {
            'datetime': datetime.datetime.now(),
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
        }
        
        for fold_id, values in folds_summary.items():
            fold_id
            Fold[1]: train(122) / test(32)
            Normalization: (mean=[0.485, 0.456, 0.406], std=[0.485, 0.456, 0.406])
            Elapsed
            time: train = 1812.54, test = 0.64
            Accuracy: 59.78 %
            Precision:
            Recall:
            F1:
            Confusion
            Matrix:
            
            
            
            

        # Append fold results
        # Append global performance

        # Substitute values
        with open(SUMMARY_TEMPLATE, 'r') as f:
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
        for fold_id in range(1, 6):
            logging.info(f'Processing fold: {fold_id}')

            # Generates fold datasets
            train_data, test_data = self._fold_handler.generate_run_set(fold_id)

            # At each iteration the model should remain the same, so conditions are equal in each fold.
            fold_model = copy(self._model)
            fold_model.to(device=self._device)

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
            fold_model, losses = train_model(model=fold_model,
                                             device=self._device,
                                             train_loader=train_data_loader,
                                             )
            tf_fold_train = time.time() - t0_fold_train

            # Test model. Test step over the train model in current fold
            # <--------------------------------------------------------------------->
            test_data_loader = load_and_transform_data(os.path.join(DYNAMIC_RUN_FOLDER, TEST), mean=custom_mean,
                                                       std=custom_std)
            # Measure test time
            t0_fold_test = time.time()
            model_performance = evaluate_model(fold_model, test_data_loader, device)
            tf_fold_test = time.time() - t0_fold_test
            model_performance
            folds_performance.append(model_performance)

            # Append fold results to summary
            self._save_fold_results()

            # Generate Loss plot
            if SAVE_LOSS_PLOT:
                self._save_plot_fold(losses, fold_id)

            # Save fold model
            if SAVE_MODEL:
                self._save_model_fold(fold_model, fold_id)

            # Run only one fold
            if MONO_FOLD:
                logging.info("Only one fold is executed")
                break

        self._save_train_summary()


        execution_time = str(timedelta(seconds=time.time() - t0))

        # # Confident interval computation
        # mean, stdev, offset, ci = get_fold_metrics(folds_acc)
        # # print(f'******************************************')
        # logging.info(f'Model performance [{execution_time}]:')
        # logging.info(f'     Folds Acc.: {folds_acc}')
        # logging.info(f'     Mean: {mean:.2f}')
        # logging.info(f'     Stdev: {stdev:.2f}')
        # logging.info(f'     Offset: {offset:.2f}')
        # logging.info(f'     CI:(95%) : [{ci[0]:.2f}, {ci[1]:.2f}]')

        # # Save model performance
        # with open(os.path.join(target_model, 'summary.out'), 'w') as f:
        #     f.write(f'Model hyperparameters::\n')
        #     f.write(f'     Batch size: {BATCH_SIZE}\n')
        #     f.write(f'     Nº epochs: {EPOCHS}\n')
        #     f.write(f'     Learning rate: {LEARNING_RATE}\n')
        #     f.write(f'     Samples Size:\n')
        #     for fold_id in range(1, 6):
        #         f.write(
        #             f'          Fold [{fold_id}]: train ({folds_samples[fold_id - 1][0]}) / test ({folds_samples[fold_id - 1][1]})\n')
        #         if custom_mean == None:
        #             f.write(f'              Normalization: Pytorch defaults\n')
        #         else:
        #             f.write(
        #                 f'              Normalization: mean=[{folds_samples[fold_id - 1][2][0]:.2f},{folds_samples[fold_id - 1][2][1]: .2f},{folds_samples[fold_id - 1][2][2]: .2f}], std=[{folds_samples[fold_id - 1][3][0]:.2f},{folds_samples[fold_id - 1][3][1]: .2f},{folds_samples[fold_id - 1][3][2]: .2f}]\n')
        #         f.write(
        #             f'              Elapsed time: train={folds_samples[fold_id - 1][4]:.2f}, test={folds_samples[fold_id - 1][5]:.2f}\n')
        #
        #     t_train = 0
        #     t_test = 0
        #     for fold_id in range(1, 6):
        #         t_train += folds_samples[fold_id - 1][4]
        #         t_test += folds_samples[fold_id - 1][5]
        #
        #     f.write(f'Model performance [{execution_time}]:\n')
        #     f.write(f'     Train = {str(timedelta(seconds=t_train))} | Test = {str(timedelta(seconds=t_test))}\n')
        #     f.write(f'     Folds Acc.: {folds_acc}\n')
        #     f.write(f'     Mean: {mean:.2f}\n')
        #     f.write(f'     Stdev: {stdev:.2f}\n')
        #     f.write(f'     Offset: {offset:.2f}\n')
        #     f.write(f'     CI:(95%) : [{ci[0]:.2f}, {ci[1]:.2f}]\n')

