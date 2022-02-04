"""
# Author: ruben 
# Date: 27/1/22
# Project: CACFramework
# File: train_constants.py

Description: constants definition related to the training stage
"""
LOG_LEVEL = 'debug'
# Train hyperparameters
# =======================

EPOCHS = 60
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 4e-2
CRITERION = 'CrossEntropyLoss'
OPTIMIZER = 'SDG'
# Architecture parameters
# =======================
ARCHITECTURE = 'efficientnet_b0'  # model architecture. Supported -->['vgg16', 'vgg19', 'resnet']
IMAGE_TYPE = 'cropped'  # Input datset type. Supported --> ['original', 'cropped']
MODEL_SEED = 3  # Fix seed to generate always deterministic results (same random numbers)
N_CLASSES = 2  # Number of classes (CAC>400, CAC<400)
CUSTOM_NORMALIZED = True  # Retrieve normalization parameters by analysing input train images

# Output parameters
# =======================
SAVE_MODEL = False  # True if model has to be saved
SAVE_LOSS_PLOT = True  # True if loss plot has to be saved
SAVE_ACCURACY_PLOT = True  # True if accuracy plot has to be saved
MONO_FOLD = False  # Run only one Fold
ND = 2  # Number of decimals at outputs
