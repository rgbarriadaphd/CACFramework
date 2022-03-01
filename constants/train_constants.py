"""
# Author: ruben 
# Date: 27/1/22
# Project: CACFramework
# File: train_constants.py

Description: constants definition related to the training stage
"""
LOG_LEVEL = 'info'
# Train hyperparameters
# =======================

EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 4e-2
CRITERION = 'CrossEntropyLoss'
OPTIMIZER = 'SDG'
# Architecture parameters
# =======================
ARCHITECTURE = "vgg16"
# model architecture supported:
# ['regnet_y_400mf','regnet_y_800mf','regnet_y_1_6gf','regnet_y_3_2gf','regnet_y_8gf','regnet_y_16gf','regnet_y_32gf',
# 'regnet_x_400mf','regnet_x_800mf','regnet_x_1_6gf','regnet_x_3_2gf','regnet_x_8gf','regnet_x_16gf','regnet_x_32gf',
# 'wide_resnet50_2','wide_resnet101_2','resnext50_32x4d','shufflenet_v2_x1_0','mnasnet0_5','mnasnet1_0','mobilenet_v2',
# 'mobilenet_v3_small','mobilenet_v3_large','shufflenet_v2_x0_5','shufflenet_v2_x1_0','googlenet','densenet121',
# 'densenet161','densenet169','densenet201','squeezenet1_1','efficientnet_b0','efficientnet_b1','efficientnet_b2',
# 'efficientnet_b3','efficientnet_b4','efficientnet_b5','efficientnet_b6','efficientnet_b7','inception_v3','alexnet',
# 'vgg16','vgg19','vgg11','vgg13','vgg16_bn','vgg19_bn','vgg11_bn','vgg13_bn','resnet18','resnet34','resnet50',
# 'resnet101','resnet152']
IMAGE_TYPE = 'cropped'  # Input datset type. Supported --> ['original', 'cropped']
MODEL_SEED = 3  # Fix seed to generate always deterministic results (same random numbers)
N_CLASSES = 2  # Number of classes (CAC>400, CAC<400)
CUSTOM_NORMALIZED = True  # Retrieve normalization parameters by analysing input train images
REQUIRES_GRAD = True  # Allow backprop in pretrained weights

# Output parameters
# =======================
SAVE_MODEL = False  # True if model has to be saved
SAVE_LOSS_PLOT = False  # True if loss plot has to be saved
SAVE_ACCURACY_PLOT = False  # True if accuracy plot has to be saved
MONO_FOLD = False  # Run only one Fold
ND = 2  # Number of decimals at outputs
