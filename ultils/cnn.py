"""
# Author: ruben 
# Date: 27/1/22
# Project: CACFramework
# File: cnn.py

Description: Functions to deal with the cnn operations
"""


import os
import numpy as np
import logging
from torchvision import models
import torch
from tqdm import tqdm
import torch.nn as nn
from torch import optim

def get_model(base_model=None, seed=False):
    """
    Gets specified architecture
    :param base_model:
    :param seed:
    :return:
    """



    '''
       Gets VGG16 model
       :param base_model: path to pre initialized model
       :return: vgg16 model with last layer modification (2 classes)
       '''
    model = models.vgg16(pretrained=True)

    if seed:
        torch.manual_seed(3)

    # Freeze trained weights
    for param in model.features.parameters():
        param.requires_grad = False

    # Newly created modules have require_grad=True by default
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1]  # Remove last layer
    linear = nn.Linear(num_features, 2)

    features.extend([linear])  # Add our layer with 2 outputs
    model.classifier = nn.Sequential(*features)  # Replace the model classifier

    # Load pre initialized model
    if base_model and os.path.exists(base_model):
        model.load_state_dict(torch.load(base_model))
        logging.info(f'Loading {base_model}')
    else:
        logging.info(f'Loading pretrained VGG16 model')

    return model


def train_model(model, device, train_loader, epochs=1, epoch_split=None, batch_size=4, lr=0.1, test_loader=None):
    n_train = len(train_loader.dataset)
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Device:          {device.type}
    ''')
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=4e-2)
    criterion = nn.CrossEntropyLoss()

    losses=[]
    for epoch in range(epochs):

        model.train(True)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for i,batch in enumerate(train_loader):
                sample, ground, file_info = batch
                sample = sample.to(device=device, dtype=torch.float32)
                ground = ground.to(device=device, dtype=torch.long)


                optimizer.zero_grad()
                prediction = model(sample)
                loss = criterion(prediction, ground)
                loss.backward()
                optimizer.step()

                if test_loader and epoch % epoch_split == 0:
                    pbar.set_postfix(**{'LR': optimizer.param_groups[0]['lr'], 'loss (batch) ': loss.item(), 'test ': acc_model})
                else:
                    pbar.set_postfix(**{'loss (batch) ': loss.item()})
                pbar.update(sample.shape[0])
        losses.append(loss.item())
    return model, losses


def evaluate_model(model, dataloader, device):
    n_test = len(dataloader.dataset)
    logging.info(f'''Starting training:
            Test size:   {n_test}
            Device:          {device.type}
        ''')
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            sample, ground, file_info = data
            sample = sample.to(device=device)
            ground = ground.to(device=device)

            outputs = model(sample)
            _, predicted = torch.max(outputs.data, 1)

            total += ground.size(0)
            correct += (predicted == ground).sum().item()
    return (100 * correct) / total