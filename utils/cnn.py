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

    def __init__(self, architecture, pretrained=True, seed=None, path=None):
        """
        Architecture class constructor
        :param architecture: (str) already existing Pytorch architecture
        :param pretrained: (bool) Whether model has to load train wights or not
        :param seed: (int) If specified, seed to generate fix random numbers
        :param path: (str) If defined, then the model has to be loaded.
        """
        logging.info(f'Loading architecture {architecture}, pretrained {pretrained}, seed {seed}')
        self._architecture = architecture
        self._pretrained = pretrained
        self._model = None
        self._path = path

        if seed:
            torch.manual_seed(seed)

        self._init()
        if self._path:
            self._model.load_state_dict(torch.load(self._path, map_location=torch.device('cpu')))
            logging.info(f'Loading {self._architecture} from {self._path}')

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
        elif self._architecture.startswith('resnext'):
            self._init_resnext()
        elif self._architecture.startswith('wide_resnet'):
            self._init_wide_resnet()
        elif self._architecture.startswith('regnet'):
            self._init_regnet()

    def _compute_weights_sum(self, architecture, model):
        """
        Compute weights sum
        :param architecture: (str) architecture name
        :param model: (torch.models) model
        """
        if architecture.startswith('vgg'):
            return model.classifier[self._index].weight.sum()
        elif architecture.startswith('resnet'):
            return self._model.fc.weight.sum()
        elif architecture.startswith('efficientnet'):
            return model.classifier[self._index].weight.sum()
        elif architecture == 'inception_v3':
            return self._model.fc.weight.sum()
        elif architecture == 'alexnet':
            return model.classifier[self._index].weight.sum()
        elif architecture == 'squeezenet1_1':
            return model.classifier[self._index].weight.sum()
        elif architecture.startswith('densenet'):
            return model.classifier.weight.sum()
        elif architecture == 'googlenet':
            return model.fc.weight.sum()
        elif architecture.startswith('shufflenet'):
            return model.fc.weight.sum()
        elif architecture.startswith('mobilenet'):
            return model.classifier[self._index].weight.sum()
        elif architecture.startswith('mnasnet'):
            return model.classifier[self._index].weight.sum()
        elif architecture.startswith('resnext'):
            return model.fc.weight.sum()
        elif architecture.startswith('wideresnet'):
            return model.fc.weight.sum()
        elif architecture.startswith('regnet'):
            return model.fc.weight.sum()

    def _init_regnet(self):
        """
        Init RegNet architecture
        """
        # Load corresponding RegNet model
        if self._architecture == 'regnet_y_400mf':
            self._model = models.regnet_y_400mf(pretrained=True)
        elif self._architecture == 'regnet_y_800mf':
            self._model = models.regnet_y_800mf(pretrained=True)
        elif self._architecture == 'regnet_y_1_6gf':
            self._model = models.regnet_y_1_6gf(pretrained=True)
        elif self._architecture == 'regnet_y_3_2gf':
            self._model = models.regnet_y_3_2gf(pretrained=True)
        elif self._architecture == 'regnet_y_8gf':
            self._model = models.regnet_y_8gf(pretrained=True)
        elif self._architecture == 'regnet_y_16gf':
            self._model = models.regnet_y_16gf(pretrained=True)
        elif self._architecture == 'regnet_y_32gf':
            self._model = models.regnet_y_32gf(pretrained=True)
        elif self._architecture == 'regnet_x_400mf':
            self._model = models.regnet_x_400mf(pretrained=True)
        elif self._architecture == 'regnet_x_800mf':
            self._model = models.regnet_x_800mf(pretrained=True)
        elif self._architecture == 'regnet_x_1_6gf':
            self._model = models.regnet_x_1_6gf(pretrained=True)
        elif self._architecture == 'regnet_x_3_2gf':
            self._model = models.regnet_x_3_2gf(pretrained=True)
        elif self._architecture == 'regnet_x_8gf':
            self._model = models.regnet_x_8gf(pretrained=True)
        elif self._architecture == 'regnet_x_16gf':
            self._model = models.regnet_x_16gf(pretrained=True)
        elif self._architecture == 'regnet_x_32gf':
            self._model = models.regnet_x_32gf(pretrained=True)

        self._modify_architecture_fc()

    def _init_wide_resnet(self):
        """
        Init Wide Resnet architecture
        """
        # Load corresponding ReNext model
        if self._architecture == 'wide_resnet50_2':
            self._model = models.wide_resnet50_2(pretrained=self._pretrained)
        elif self._architecture == 'wide_resnet101_2':
            self._model = models.wide_resnet101_2(pretrained=self._pretrained)

        self._modify_architecture_fc()

    def _init_resnext(self):
        """
        Init ResNext architecture
        """
        # Load corresponding ResNext model
        if self._architecture == 'resnext50_32x4d':
            self._model = models.resnext50_32x4d(pretrained=self._pretrained)
        elif self._architecture == 'shufflenet_v2_x1_0':
            self._model = models.resnext101_32x8d(pretrained=self._pretrained)

        self._modify_architecture_fc()

    def _init_mnasnet(self):
        """
        Init Mobilenet architecture
        """
        # Load corresponding mobilenet model
        if self._architecture == 'mnasnet0_5':
            self._model = models.mnasnet0_5(pretrained=self._pretrained)
        elif self._architecture == 'mnasnet1_0':
            self._model = models.mnasnet1_0(pretrained=self._pretrained)

        self._index = 1
        self._modify_architecture_classifier()

    def _init_mobilenet(self):
        """
        Init Mobilenet architecture
        """
        # Load corresponding mobilenet model
        if self._architecture == 'mobilenet_v2':
            self._model = models.mobilenet_v2(pretrained=self._pretrained)
            self._index = 1
        elif self._architecture == 'mobilenet_v3_small':
            self._model = models.mobilenet_v3_small(pretrained=self._pretrained)
            self._index = 3
        elif self._architecture == 'mobilenet_v3_large':
            self._model = models.mobilenet_v3_large(pretrained=self._pretrained)
            self._index = 3

        self._modify_architecture_classifier()

    def _init_shufflenet(self):
        """
        Init Shufflenet architecture
        """
        # Load corresponding Shufflenet model
        if self._architecture == 'shufflenet_v2_x0_5':
            self._model = models.shufflenet_v2_x0_5(pretrained=self._pretrained)
        elif self._architecture == 'shufflenet_v2_x1_0':
            self._model = models.shufflenet_v2_x1_0(pretrained=self._pretrained)

        self.modify_architecture_fc()

    def _init_googlenet(self):
        """
        Init Googlenet architecture
        """
        # Load corresponding googlenet model
        if self._architecture == 'googlenet':
            self._model = models.googlenet(pretrained=self._pretrained)

        self._modify_architecture_fc()

    def _init_densenet(self):
        """
        Init DenseNet architecture
        """
        # Load corresponding vgg model
        if self._architecture == 'densenet121':
            self._model = models.densenet121(pretrained=self._pretrained)
        elif self._architecture == 'densenet161':
            self._model = models.densenet161(pretrained=self._pretrained)
        elif self._architecture == 'densenet169':
            self._model = models.densenet169(pretrained=self._pretrained)
        elif self._architecture == 'densenet201':
            self._model = models.densenet201(pretrained=self._pretrained)

        # Freeze trained weights
        for param in self._model.features.parameters():
            param.requires_grad = REQUIRES_GRAD

        # Adapt architecture. Newly created modules have require_grad=True by default
        num_features = self._model.classifier.in_features
        self._model.classifier = nn.Linear(num_features, N_CLASSES)

        # Base weights sum
        self._weights_sum = self._model.classifier.weight.sum()

    def _init_squeezenet(self):
        """
        Init Squeezenet 1.1 architecture
        """
        # Load corresponding vgg model
        if self._architecture == 'squeezenet1_1':
            self._model = models.squeezenet1_1(pretrained=self._pretrained)
        # TBI: rest of squeeze architectures. worth?

        # Freeze trained weights
        for param in self._model.features.parameters():
            param.requires_grad = REQUIRES_GRAD

        # Adapt architecture. Newly created modules have require_grad=True by default
        self._model.classifier[1] = nn.Conv2d(512, N_CLASSES, kernel_size=(1, 1), stride=(1, 1))

        # Base weights sum
        self._weights_sum = self._model.classifier[1].weight.sum()

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

        self._index = 1
        self._modify_architecture_classifier()

    def _init_inception_v3(self):
        """
        Init Inception v3 architecture
        """
        # Load corresponding vgg model
        if self._architecture == 'inception_v3':
            self._model = models.inception_v3(pretrained=self._pretrained)

        # FIXME
        # Freeze trained weights
        for param in self._model.parameters():
            param.requires_grad = REQUIRES_GRAD

        # Handle the auxilary net
        num_ftrs = self._model.AuxLogits.fc.in_features
        self._model.AuxLogits.fc = nn.Linear(num_ftrs, N_CLASSES)
        # Handle the primary net
        num_ftrs = self._model.fc.in_features
        self._model.fc = nn.Linear(num_ftrs, N_CLASSES)
        input_size = 299

        # Adapt architecture. Newly created modules have require_grad=True by default
        aux_n_features = self._model.AuxLogits.fc.in_features
        self._model.AuxLogits.fc = nn.Linear(aux_n_features, N_CLASSES)

        num_features = self._model.fc.in_features
        self._model.fc = nn.Linear(num_features, N_CLASSES)

        # Base weights sum
        self._weights_sum = self._model.fc.weight.sum()

    def _init_alexnet(self):
        """
        Init Alexnet architecture
        """
        # Load corresponding vgg model
        if self._architecture == 'alexnet':
            self._model = models.alexnet(pretrained=self._pretrained)

        self._index = 6
        self._modify_architecture_classifier()

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

        self._index = 6
        self._modify_architecture_classifier()

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

        self._modify_architecture_fc()

    def _modify_architecture_fc(self):

        # Freeze trained weights
        for param in self._model.parameters():
            param.requires_grad = REQUIRES_GRAD

        # Adapt architecture. Newly created modules have require_grad=True by default
        num_features = self._model.fc.in_features
        linear = nn.Linear(num_features, N_CLASSES)
        self._model.fc = linear  # Replace the model classifier

        # Base weights sum
        self._weights_sum = self._model.fc.weight.sum()

    def _modify_architecture_classifier(self):

        # Freeze trained weights
        for param in self._model.features.parameters():
            param.requires_grad = REQUIRES_GRAD

        # Adapt architecture. Newly created modules have require_grad=True by default
        num_features = self._model.classifier[self._index].in_features
        features = list(self._model.classifier.children())[:-1]  # Remove last layer
        linear = nn.Linear(num_features, N_CLASSES)

        features.extend([linear])  # Add our layer with 2 outputs
        self._model.classifier = nn.Sequential(*features)  # Replace the model classifier

        # Base weights sum
        self._weights_sum = self._model.classifier[self._index].weight.sum()

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

    def compute_weights_external(self, architecture, model):
        """
        Giving an external model, compute corresponding weights sum
        :param architecture: (str) architecture name
        :param model: (torch.models) model
        """
        return self._compute_weights_sum(architecture, model)


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
                if ARCHITECTURE == 'inception_v3':
                    prediction = prediction.logits
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
    logging.info(f'''Starting tesing:
            Test size:  {n_test}
            Device:     {device.type}
            Fold ID:    {fold_id}
        ''')

    correct = 0
    total = 0
    ground_array = []
    prediction_array = []
    dataset_info = {}
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            sample, ground, file_info = data
            sample = sample.to(device=device)
            ground = ground.to(device=device)

            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)

            sample_name = file_info[0][0].split('/')[-1].split('.')[0]
            dataset_info[sample_name] = {'ground': None, 'prediction': None}
            assert ground.item() == file_info[1][0]
            dataset_info[sample_name]['ground'] = ground.item()
            dataset_info[sample_name]['prediction'] = predicted.item()

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
           }, (100 * correct) / total, dataset_info
