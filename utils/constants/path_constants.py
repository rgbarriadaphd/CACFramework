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

ROOT_ORIGINAL_CROP_FOLDS_NERVE = 'input/image/crop_folds_nerve'
assert os.path.exists(ROOT_ORIGINAL_CROP_FOLDS_NERVE)

DYNAMIC_RUN_FOLDER = 'input/image/dynamic_run'
assert os.path.exists(DYNAMIC_RUN_FOLDER)

MODELS_FOLDER = 'models'
assert os.path.exists(MODELS_FOLDER)

TEMPLATES_FOLDER = 'templates'
assert os.path.exists(TEMPLATES_FOLDER)

OUTPUT_FOLDER = 'output'
assert os.path.exists(OUTPUT_FOLDER)

LOGS_FOLDER = 'logs'
assert os.path.exists(LOGS_FOLDER)

TRAIN = 'train'
TEST = 'test'
CAC_NEGATIVE = 'CACSmenos400'
CAC_POSITIVE = 'CACSmas400'


# Templates

SUMMARY_TEMPLATE_ALL = 'templates/summary.template'
assert os.path.exists(SUMMARY_TEMPLATE_ALL)

SUMMARY_TEMPLATE_FOLD_1 = 'templates/summary_f1.template'
assert os.path.exists(SUMMARY_TEMPLATE_FOLD_1)

SUMMARY_TEMPLATE_FOLD_2 = 'templates/summary_f2.template'
assert os.path.exists(SUMMARY_TEMPLATE_FOLD_2)

SUMMARY_TEMPLATE_FOLD_3 = 'templates/summary_f3.template'
assert os.path.exists(SUMMARY_TEMPLATE_FOLD_3)

SUMMARY_TEMPLATE_FOLD_4 = 'templates/summary_f4.template'
assert os.path.exists(SUMMARY_TEMPLATE_FOLD_4)

SUMMARY_TEMPLATE_FOLD_5 = 'templates/summary_f5.template'
assert os.path.exists(SUMMARY_TEMPLATE_FOLD_5)

