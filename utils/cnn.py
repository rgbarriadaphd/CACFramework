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
from utils.metrics import PerformanceMetrics


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
        if self._architecture == 'vgg16':
            self._init_vgg('vgg16')
        elif self._architecture == 'vgg19':
            self._init_vgg('vgg19')
        elif self._architecture == 'resnet18':
            self._init_resnet('resnet18')

    def _init_vgg(self, version='vgg16'):
        """
        Init architecture
        :param version: (str) vgg version. vgg16 by default
        """
        # Load corresponding vgg model
        if version == 'vgg16':
            self._model = models.vgg16(pretrained=self._pretrained)
        elif version == 'vgg19':
            # TBI: implement vgg versions
            pass

        # TODO: verify version compatibility

        # Freeze trained weights
        for param in self._model.features.parameters():
            param.requires_grad = False

        # Adapt architecture. Newly created modules have require_grad=True by default
        num_features = self._model.classifier[6].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, N_CLASSES)

        features.extend([linear])  # Add our layer with 2 outputs
        self._model.classifier = nn.Sequential(*features)  # Replace the model classifier

    def get(self):
        return self._model


def train_model(model, device, train_loader):
    """
    Trains the model with input parametrization
    :param model: (torchvision.models) Pytorch model
    :param device: (torch.cuda.device) Computing device
    :param train_loader: (torchvision.datasets) Train dataloader containing dataset images
    :return: train model
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
    return model, losses


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


