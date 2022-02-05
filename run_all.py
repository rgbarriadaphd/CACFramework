"""
# Author: ruben 
# Date: 4/2/22
# Project: CACFramework
# File: run_all.py

Description: "Enter feature description here"
"""
from constants.train_constants import ARCHITECTURE

import os

ARCHITECTURES = ['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf',
                 'regnet_y_16gf', 'regnet_y_32gf', 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf',
                 'regnet_x_3_2gf', 'regnet_x_8gf', 'regnet_x_16gf', 'regnet_x_32gf', 'wide_resnet50_2',
                 'wide_resnet101_2', 'resnext50_32x4d', 'shufflenet_v2_x1_0', 'mnasnet0_5', 'mnasnet1_0',
                 'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
                 'googlenet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'squeezenet1_1',
                 'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
                 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7', 'inception_v3', 'alexnet', 'vgg16', 'vgg19',
                 'vgg11', 'vgg13', 'vgg16_bn', 'vgg19_bn', 'vgg11_bn', 'vgg13_bn', 'resnet18', 'resnet34', 'resnet50',
                 'resnet101', 'resnet152']


def modify_input_architecture(model):
    constants_file = 'constants/train_constants.py'
    with open(constants_file, 'r') as f:
        s = f.read()
        print('ARCHITECTURE =' in s)



def main():
    modify_input_architecture('resnet152')
    os.system("python cac_main.py -h")



if __name__ == '__main__':
    main()
