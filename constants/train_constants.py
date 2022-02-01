"""
# Author: ruben 
# Date: 27/1/22
# Project: CACFramework
# File: train_constants.py

Description: constants definition related to the training stage
"""

# Train hyperparameters
# =======================

EPOCHS = 1
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 4e-2
CRITERION = 'CrossEntropyLoss'
OPTIMIZER = 'SDG'
# Architecture parameters
# =======================
ARCHITECTURE = 'vgg16'  # model architecture. Supported -->['vgg16', 'vgg19', 'resnet']
IMAGE_TYPE = 'cropped'  # Input datset type. Supported --> ['original', 'cropped']
MODEL_SEED = True  # Fix seed to generate always deterministic results (same random numbers)
N_CLASSES = 2  # Number of classes (CAC>400, CAC<400)
CUSTOM_NORMALIZED = True  # Retrieve normalization parameters by analysing input train images

# Output parameters
# =======================
SAVE_MODEL = True  # True if model has to be saved
SAVE_LOSS_PLOT = True  # True if loss plot has to be saved
MONO_FOLD = False  # Run only one Fold
