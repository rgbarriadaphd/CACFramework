"""
# Author: ruben
# Date: 26/1/22
# Project: CACFramework
# File: model_train.py

Description: Class to handle train stages
"""
import logging
import torch
import datetime

from constants.path_constants import *
from constants.train_constants import *

from ultils.fold_handler import FoldHandler


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

        logging.info(f'{self._architecture} | {self._images}')

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
        self._model = get_model(self._architecture, seed=MODEL_SEED)

    def _get_normalization(self):
        # Generate custom mean and std normalization values from only train dataset
        if CUSTOM_NORMALIZED:
            logging.info(f'Custom image normalization')
            mean, std = get_custom_normalization()
        else:
            logging.info(f'Pytorch image normalization')
            mean = None
            std = None
        return mean, std

    def run(self):


        t0 = time.time()
        folds_acc = []
        folds_samples = []
        for fold_id in range(1, 6):
            logging.info(f'Processing fold: {fold_id}')

            # Generates fold datasets
            train_data, test_data = self._fold_handler.generate_run_set(fold_id)

            # At each iteration the model should remain the same, so conditions are equal in each fold.
            fold_model = copy(model)
            fold_model.to(device=device)

            custom_mean, custom_std = self._get_normalization()

            train_data_loader = load_and_transform_data(os.path.join(DYNAMIC_RUN_FOLDER, TRAIN),
                                                        batch_size=BATCH_SIZE,
                                                        data_augmentation=False, mean=custom_mean, std=custom_std)
            t0_fold_train = time.time()
            # train model
            fold_model, losses = train_model(model=fold_model,
                                             device=device,
                                             train_loader=train_data_loader,
                                             epochs=EPOCHS,
                                             batch_size=BATCH_SIZE,
                                             lr=LEARNING_RATE,
                                             test_loader=None)

            tf_fold_train = time.time() - t0_fold_train

            # # test model
            # test_data_loader = load_and_transform_data(os.path.join(DYNAMIC_RUN_FOLDER, TEST), mean=custom_mean,
            #                                            std=custom_std)
            #
            # logging.info("Test model after training")
            #
            # t0_fold_test = time.time()
            # acc_model = evaluate_model(fold_model, test_data_loader, device)
            # tf_fold_test = time.time() - t0_fold_test
            # folds_acc.append(acc_model)
            #
            # logging.info(f'Accuracy after training {acc_model}. | [{time.time() - t0:0.3f}]')
            #
            # folds_samples.append((
            #                      len(train_data_loader.dataset), len(test_data_loader.dataset), custom_mean, custom_std,
            #                      tf_fold_train, tf_fold_test))

            # Save fold model
            folder_model_path = os.path.join(target_model, f'model_folder_{fold_id}.pt')
            logging.info(f'Saving folder model to {folder_model_path}')
            torch.save(fold_model.state_dict(), folder_model_path)

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

