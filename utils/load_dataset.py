"""
# Author: ruben 
# Date: 29/1/22
# Project: CACFramework
# File: load_dataset.py

Description: Functions to load and transform input images
"""
import os
from PIL import Image, ImageStat
import numpy as np
import logging
import torch
from torchvision import datasets, transforms
from typing import Dict, List, Tuple

from constants.path_constants import DYNAMIC_RUN_FOLDER, TRAIN


def get_custom_normalization():
    """
    Get normalization according to input train dataset
    :return: ((list) mean, (list) std) Normalization values mean and std
    """
    target = os.path.join(DYNAMIC_RUN_FOLDER, TRAIN)

    means = []
    stds = []

    for root, dirs, files in os.walk(target):
        for file in files:
            image_path = os.path.join(root, file)
            assert os.path.exists(image_path)
            with open(image_path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                stat = ImageStat.Stat(img)
                local_mean = [stat.mean[0] / 255., stat.mean[1] / 255., stat.mean[2] / 255.]
                local_std = [stat.stddev[0] / 255., stat.stddev[1] / 255., stat.stddev[2] / 255.]
                means.append(local_mean)
                stds.append(local_std)

    return list(np.array(means).mean(axis=0)), list(np.array(stds).mean(axis=0))


class CustomImageFolder(datasets.ImageFolder):
    """
    Custom ImageFolder class. Workaround to swap class index assignment.
    """
    def __init__(self, dataset, transform=None):
        """

        :param dataset: (str) Dataset path
        :param transform: (torch.transforms) Set of transforms to be applied to input data
        """
        super(CustomImageFolder, self).__init__(dataset, transform=transform)
        self.class_to_idx = {'CACSmenos400': 0, 'CACSmas400': 1}


def load_and_transform_data(dataset, batch_size=1, data_augmentation=False, mean=None, std=None):
    """
    Loads a dataset and applies the corresponding transformations
    :param dataset:
    :param batch_size:
    :param data_augmentation:
    :param mean: (list) Normalized mean
    :param std: (list) Normalized std
    """
    # Define transformations that will be applied to the images
    # VGG-16 Takes 224x224 images as input, so we resize all of them
    logging.info(f'Loading data from {dataset}')

    if mean is None and std is None:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        logging.info(f'Applying custom normalization: mean={mean}, std={std}')


    if data_augmentation:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)), # FIXME: architecture dependant parametrization
            transforms.RandomRotation(20),
            transforms.RandomRotation(110),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)), # FIXME: architecture dependant parametrization
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    image_datasets = CustomImageFolder(dataset, transform=data_transforms)
    im = Image.open(image_datasets.imgs[0][0])
    logging.info(f'Sample size: {im.size}')

    data_loader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True, num_workers=4)

    logging.info(f'Loaded {len(image_datasets)} images under {dataset}: Classes: {image_datasets.class_to_idx}')

    return data_loader