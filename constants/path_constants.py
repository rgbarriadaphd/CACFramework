"""
# Author: ruben 
# Date: 27/1/22
# Project: CACFramework
# File: path_constants.py

Description: Constants related to project paths. The have to exists since the begining
"""

import os

# PATHS
CLINICAL_DATA = 'input/clinical/base_clinical_data.xlsx'
assert os.path.exists(CLINICAL_DATA)

ROOT_ORIGINAL_FOLDS = 'input/image/folds'
assert os.path.exists(ROOT_ORIGINAL_FOLDS)

ROOT_ORIGINAL_CROP_FOLDS = 'input/image/crop_folds'
assert os.path.exists(ROOT_ORIGINAL_CROP_FOLDS)

DYNAMIC_RUN_FOLDER = 'input/image/dynamic_run'
assert os.path.exists(DYNAMIC_RUN_FOLDER)

MODELS_FOLDER = 'models'
assert os.path.exists(MODELS_FOLDER)

LOGS_FOLDER = 'logs'
assert os.path.exists(LOGS_FOLDER)

TRAIN = 'train'
TEST = 'test'
CAC_NEGATIVE = 'CACSmenos400'
CAC_POSITIVE = 'CACSmas400'


# Templates

SUMMARY_TEMPLATE = 'templates/summary.template'
assert os.path.exists(SUMMARY_TEMPLATE)
