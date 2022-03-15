"""
# Author: ruben 
# Date: 4/2/22
# Project: CACFramework
# File: run_all.py

Description: Script to train all available architectures over the train dataset
"""
import os
import sys

# ARCHITECTURES = ['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf',
#                  'regnet_y_16gf', 'regnet_y_32gf',
#                  'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_8gf',
#                  'regnet_x_16gf', 'regnet_x_32gf',
#                  'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'shufflenet_v2_x1_0', 'mnasnet0_5',
#                  'mnasnet1_0', 'mobilenet_v2',
#                  'mobilenet_v3_small', 'mobilenet_v3_large', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'googlenet',
#                  'densenet121',
#                  'densenet161', 'densenet169', 'densenet201', 'squeezenet1_1', 'efficientnet_b0', 'efficientnet_b1',
#                  'efficientnet_b2',
#                  'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
#                  'inception_v3', 'alexnet',
#                  'vgg16', 'vgg19', 'vgg11', 'vgg13', 'vgg16_bn', 'vgg19_bn', 'vgg11_bn', 'vgg13_bn', 'resnet18',
#                  'resnet34', 'resnet50',
#                  'resnet101', 'resnet152']


ARCHITECTURES = ['regnet_x_8gf', 'inception_v3','vgg16','resnet18']

def modify_input_architecture(model):
    """
    Change the architecture definition of the train_constants.py file before run
    :param model: (str) Current architecture
    """
    constants_file = 'constants/train_constants.py'
    new_lines = []
    with open(constants_file, 'r') as f_in:
        for line in f_in.readlines():
            if line.startswith('ARCHITECTURE'):
                new_lines.append(f'ARCHITECTURE = "{model}"\n')
            else:
                new_lines.append(line)
    with open(constants_file, 'w') as f_out:
        f_out.writelines(new_lines)


def get_model_parameters(model, model_folder='models'):
    """
    Retrieve the performance parameters from the summary of a specific model
    :param model_folder: (str) input path for models folder
    :param model: (str) Current architecture
    """
    folds_acc, mean, stddev, interval = None, None, None, None
    print(model_folder)
    for folder in os.listdir(model_folder):
        if f'train_{model}' in folder:
            summary_file = os.path.join(model_folder, folder, 'summary.out')

            assert os.path.exists(summary_file), f'{summary_file} not found'
            with open(summary_file, 'r') as f_in:
                for line in f_in.readlines():
                    if line.startswith('Folds Acc.'):
                        folds_acc = line.split(':')[1].strip()
                    if line.startswith('Mean'):
                        mean = line.split(':')[1].strip()
                    if line.startswith('StdDev'):
                        stddev = line.split(':')[1].strip()
                    if line.startswith('CI:(95%)'):
                        interval = line.split(':')[2].strip()

    return folds_acc, mean, stddev, interval


def write_latex_table(model_parameters, output_file='templates/models.latex'):
    """
    Write a latex table format with all models performance
    :param output_file: (str) path to output file
    :param model_parameters: (dict) Contains model performance by mdoel
    """
    with open(output_file, 'w') as f_out:
        f_out.write('\\begin{table}[H]\n')
        f_out.write('\caption{Python architecture main models performance.}\n')
        f_out.write('\centering\n')
        f_out.write('\\begin{tabular}{c|c c c | c }\n')
        f_out.write(
            '\t\\textbf{Model} & \\textbf{Folds Accuracy} & \\textbf{Mean} & \\textbf{Std. Dev} & \\textbf{CI ($95\%$)}\\\\ \n')
        f_out.write('\t\hline\n')
        f_out.write('\t\hline\n')

        for model, param in model_parameters.items():
            formatted = model
            if '_' in model:
                formatted = formatted.replace('_', '\_')

            f_out.write(f'\t{formatted} & {param[0]} & {param[1]} & {param[2]} & {param[3]}\\\\ \n')
            f_out.write(f'\t\hline\n')

        f_out.write('\end{tabular}\n')
        f_out.write('\label{tab:pythorch_models}\n')
        f_out.write('\end{table}\n')


def is_computed(model):
    """
    Check if a model has already been executed
    :param model: (model) The model
    :return: True if the model has been computed, False otherwise
    """
    for folder in os.listdir('models'):
        if f'train_{model}' in folder:
            summary_file = os.path.join('models', folder, 'summary.out')
            if os.path.exists(summary_file):
                return True
    return False

def run_all(args):
    models_parameters = dict.fromkeys(ARCHITECTURES)

    #TODO: rearrange and think options more robust!!!
    if len(args) == 3:
        model_folder = args[1].split('=')[1]
        out_file = args[2].split('=')[1]
        for model in ARCHITECTURES:
            models_parameters[model] = get_model_parameters(model, model_folder)
        write_latex_table(models_parameters, out_file)
    else:
        for model in ARCHITECTURES:
            if not is_computed(model):
                print(f'{model} not computed, launch train process')
                if len(args) == 1:
                    modify_input_architecture(model)
                    os.system("python cac_main.py -tr")
                models_parameters[model] = get_model_parameters(model)
            else:
                print(f'{model} already computed')
        write_latex_table(models_parameters)


if __name__ == '__main__':
    run_all(sys.argv)
