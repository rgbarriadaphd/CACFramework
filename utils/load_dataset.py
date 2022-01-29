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