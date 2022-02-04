"""
# Author: ruben 
# Date: 27/1/22
# Project: CACFramework
# File: cnn.py

Description: Functions to deal with the cnn operations
"""

import os
import logging
import logging
from torchvision import models
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim

from constants.train_constants import *
from constants.path_constants import *
from utils.metrics import PerformanceMetrics
from utils.load_dataset import load_and_transform_data


class Architecture:
    """
    Class to manage the architecture initialization
    """

    def __init__(self, architecture, pretrained=True, seed=None):
        """
        Architecture class constructor
        :param architecture: (str) already existing Pytorch architecture
        :param pretrained: (bool) Whether model has to load train wights or not
        :param seed: (int) If specified, seed to generate fix random numbers
        """
        logging.info(f'Loading architecture {architecture}, pretrained {pretrained}, seed {seed}')
        self._architecture = architecture
        self._pretrained = pretrained
        self._model = None

        if seed:
            torch.manual_seed(seed)

        self._init()

    def _init(self):
        """
        Initialize model architecture
        """
        if self._architecture.startswith('vgg'):
            self._init_vgg()
        elif self._architecture.startswith('resnet'):
            self._init_resnet()
        elif self._architecture.startswith('efficientnet'):
            self._init_efficientnet()
        elif self._architecture == 'inception_v3':
            self._init_inception_v3()
        elif self._architecture == 'alexnet':
            self._init_alexnet()
        elif self._architecture == 'squeezenet1_1':
            self._init_squeezenet()
        elif self._architecture.startswith('densenet'):
            self._init_densenet()
        elif self._architecture == 'googlenet':
            self._init_googlenet()
        elif self._architecture.startswith('shufflenet'):
            self._init_shufflenet()
        elif self._architecture.startswith('mobilenet'):
            self._init_mobilenet()
        elif self._architecture.startswith('mnasnet'):
            self._init_mnasnet()

    def _compute_weights_sum(self, model):
        """
        Compute weights sum
        :param model: (torch.models) model
        """
        if self._architecture.startswith('vgg'):
            return model.classifier[6].weight.sum()
        elif self._architecture.startswith('resnet'):
            return self._model.fc.weight.sum()

    def _init_efficientnet(self):
        """
        Init Efficient architecture
        """
        # Load corresponding vgg model
        if self._architecture == 'efficientnet_b0':
            self._model = models.efficientnet_b0(pretrained=self._pretrained)
        elif self._architecture == 'efficientnet_b1':
            self._model = models.efficientnet_b1(pretrained=self._pretrained)
        elif self._architecture == 'efficientnet_b2':
            self._model = models.efficientnet_b2(pretrained=self._pretrained)
        elif self._architecture == 'efficientnet_b3':
            self._model = models.efficientnet_b3(pretrained=self._pretrained)
        elif self._architecture == 'efficientnet_b4':
            self._model = models.efficientnet_b4(pretrained=self._pretrained)
        elif self._architecture == 'efficientnet_b5':
            self._model = models.efficientnet_b5(pretrained=self._pretrained)
        elif self._architecture == 'efficientnet_b6':
            self._model = models.efficientnet_b6(pretrained=self._pretrained)
        elif self._architecture == 'efficientnet_b7':
            self._model = models.efficientnet_b7(pretrained=self._pretrained)

        # FIXME
        # Freeze trained weights
        for param in self._model.features.parameters():
            param.requires_grad = False

        # Adapt architecture. Newly created modules have require_grad=True by default
        num_features = self._model.classifier[6].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, N_CLASSES)

        features.extend([linear])  # Add our layer with 2 outputs
        self._model.classifier = nn.Sequential(*features)  # Replace the model classifier

        # Base weights sum
        self._weights_sum = self._model.classifier[6].weight.sum()

    def _init_inception_v3(self):
        """
        Init Inception v3 architecture
        """
        # Load corresponding vgg model
        if self._architecture == 'inception_v3':
            self._model = models.inception_v3(pretrained=self._pretrained)

        # FIXME
        # Freeze trained weights
        for param in self._model.features.parameters():
            param.requires_grad = False

        # Adapt architecture. Newly created modules have require_grad=True by default
        num_features = self._model.classifier[6].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, N_CLASSES)

        features.extend([linear])  # Add our layer with 2 outputs
        self._model.classifier = nn.Sequential(*features)  # Replace the model classifier

        # Base weights sum
        self._weights_sum = self._model.classifier[6].weight.sum()

    def _init_vgg(self):
        """
        Init VGG architecture
        """
        # Load corresponding vgg model
        if self._architecture == 'vgg16':
            self._model = models.vgg16(pretrained=self._pretrained)
        elif self._architecture == 'vgg19':
            self._model = models.vgg19(pretrained=self._pretrained)
        elif self._architecture == 'vgg11':
            self._model = models.vgg11(pretrained=self._pretrained)
        elif self._architecture == 'vgg13':
            self._model = models.vgg13(pretrained=self._pretrained)
        elif self._architecture == 'vgg16_bn':
            self._model = models.vgg16_bn(pretrained=self._pretrained)
        elif self._architecture == 'vgg19_bn':
            self._model = models.vgg19_bn(pretrained=self._pretrained)
        elif self._architecture == 'vgg11_bn':
            self._model = models.vgg11_bn(pretrained=self._pretrained)
        elif self._architecture == 'vgg13_bn':
            self._model = models.vgg13_bn(pretrained=self._pretrained)

        # Freeze trained weights
        for param in self._model.features.parameters():
            param.requires_grad = False

        # Adapt architecture. Newly created modules have require_grad=True by default
        num_features = self._model.classifier[6].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, N_CLASSES)

        features.extend([linear])  # Add our layer with 2 outputs
        self._model.classifier = nn.Sequential(*features)  # Replace the model classifier

        # Base weights sum
        self._weights_sum = self._model.classifier[6].weight.sum()

    def _init_resnet(self):
        """
        Init Resnet architecture
        """
        # Load corresponding vgg model
        if self._architecture == 'resnet18':
            self._model = models.resnet18(pretrained=self._pretrained)
        elif self._architecture == 'resnet34':
            self._model = models.resnet34(pretrained=self._pretrained)
        elif self._architecture == 'resnet50':
            self._model = models.resnet50(pretrained=self._pretrained)
        elif self._architecture == 'resnet101':
            self._model = models.resnet101(pretrained=self._pretrained)
        elif self._architecture == 'resnet152':
            self._model = models.resnet152(pretrained=self._pretrained)

        # Freeze trained weights
        for param in self._model.parameters():
            param.requires_grad = False

        # Adapt architecture. Newly created modules have require_grad=True by default
        num_features = self._model.fc.in_features
        linear = nn.Linear(num_features, N_CLASSES)
        self._model.fc = linear  # Replace the model classifier

        # Base weights sum
        self._weights_sum = self._model.fc.weight.sum()

    def get(self):
        """
        Return model
        """
        return self._model

    def weights_sum(self):
        """
        Return last layer weights sum
        """
        return self._weights_sum

    def compute_weights_external(self, model):
        """
        Giving a external model, compute corresponding weights sum
        :param model: (torch.models) model
        """
        return self._compute_weights_sum(model)


def train_model(model, device, train_loader, normalization=None):
    """
    Trains the model with input parametrization
    :param model: (torchvision.models) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param train_loader: (torchvision.datasets) Train dataloader containing dataset images
    :param normalization: Normalization to test train dataset
    :return: train model, losses array, accuracies of test and train datasets
    """
    n_train = len(train_loader.dataset)
    logging.info(f'''Starting training:
        Epochs:          {EPOCHS}
        Batch size:      {BATCH_SIZE}
        Learning rate:   {LEARNING_RATE}
        Training size:   {n_train}
        Device:          {device.type}
        Criterion:       {CRITERION}
        Optimizer:       {OPTIMIZER}
    ''')
    # TODO: Parametrize optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # TODO: Parametrize loss convergence function
    losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(EPOCHS):
        model.train(True)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{EPOCHS}', unit='img') as pbar:
            for i, batch in enumerate(train_loader):
                sample, ground, file_info = batch
                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)

                optimizer.zero_grad()
                prediction = model(sample)
                loss = criterion(prediction, ground)
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{'loss (batch) ': loss.item()})
                pbar.update(sample.shape[0])
        losses.append(loss.item())

        if SAVE_ACCURACY_PLOT:
            logging.info('Evaluate train/test dataset accuracy')

            for data_element in [TRAIN, TEST]:
                data_loader = load_and_transform_data(os.path.join(DYNAMIC_RUN_FOLDER, data_element),
                                                            mean=normalization[0],
                                                            std=normalization[1])
                _, accuracy = evaluate_model(model, data_loader, device, None)
                if data_element == TRAIN:
                    train_accuracies.append(accuracy)
                else:
                    test_accuracies.append(accuracy)
    return model, losses, (train_accuracies, test_accuracies)


def evaluate_model(model, test_loader, device, fold_id):
    """
    Test the model with input parametrization
    :param model: (torch) Pytorch model
    :param test_loader: (torchvision.datasets) Test dataloader containing dataset images
    :param device: (torch.cuda.device) Computing device
    :param fold_id: (int) Fold identifier. Just to return data.
    :return: (dict) model performance including accuracy, precision, recall, F1-measure
            and confusion matrix
    """
    n_test = len(test_loader.dataset)
    logging.info(f'''Starting training:
            Test size:  {n_test}
            Device:     {device.type}
            Fold ID:    {fold_id}
        ''')

    correct = 0
    total = 0
    ground_array = []
    prediction_array = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            sample, ground, file_info = data
            sample = sample.to(device=device)
            ground = ground.to(device=device)

            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)

            ground_array.append(ground.item())
            prediction_array.append(predicted.item())

            total += ground.size(0)
            correct += (predicted == ground).sum().item()

    pm = PerformanceMetrics(ground=ground_array,
                            prediction=prediction_array,
                            percent=True,
                            formatted=True)
    confusion_matrix = pm.confusion_matrix()

    return {
               f'accuracy_{fold_id}': pm.accuracy(),
               f'precision_{fold_id}': pm.precision(),
               f'recall_{fold_id}': pm.recall(),
               f'f1_{fold_id}': pm.f1(),
               f'tn_{fold_id}': confusion_matrix[0],
               f'fp_{fold_id}': confusion_matrix[1],
               f'fn_{fold_id}': confusion_matrix[2],
               f'tp_{fold_id}': confusion_matrix[3]
           }, (100 * correct) / total
