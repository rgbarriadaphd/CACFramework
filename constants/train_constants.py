"""
# Author: ruben 
# Date: 27/1/22
# Project: CACFramework
# File: train_constants.py

Description: constants definition related to the training stage
"""

# Train hyperparameters
# =======================

EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 4e-2

# Architecture parameters
# =======================
MODEL_SEED = True  # Fix seed to generate always deterministic results (same random numbers)
N_CLASSES = 2  # Number of classes (CAC>400, CAC<400)
CUSTOM_NORMALIZED = True  # Retrieve normalization parameters by analysing input train images
