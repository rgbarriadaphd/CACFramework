"""
# Author: ruben
# Date: 4/2/22
# Project: CACFramework
# File: run_all.py

Description: Script to train all available architectures over the train dataset
"""
import os
import sys

ARCHITECTURES = ['regnet_y_400mf', 'regnet_y_800mf', 'regnet_y_1_6gf', 'regnet_y_3_2gf', 'regnet_y_8gf',
                 'regnet_y_16gf', 'regnet_y_32gf',
                 'regnet_x_400mf', 'regnet_x_800mf', 'regnet_x_1_6gf', 'regnet_x_3_2gf', 'regnet_x_8gf',
                 'regnet_x_16gf', 'regnet_x_32gf',
                 'wide_resnet50_2', 'wide_resnet101_2', 'resnext50_32x4d', 'shufflenet_v2_x1_0', 'mnasnet0_5',
                 'mnasnet1_0', 'mobilenet_v2',
                 'mobilenet_v3_small', 'mobilenet_v3_large', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'googlenet',
                 'densenet121',
                 'densenet161', 'densenet169', 'densenet201', 'squeezenet1_1', 'efficientnet_b0', 'efficientnet_b1',
                 'efficientnet_b2',
                 'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7',
                 'inception_v3', 'alexnet',
                 'vgg16', 'vgg19', 'vgg11', 'vgg13', 'vgg16_bn', 'vgg19_bn', 'vgg11_bn', 'vgg13_bn', 'resnet18',
                 'resnet34', 'resnet50',
                 'resnet101', 'resnet152']


def get_model_parameters(model, model_folder='models'):
    """
    Retrieve the performance parameters from the summary of a specific model
    :param model_folder: (str) input path for models folder
    :param model: (str) Current architecture
    """
    mean, stddev = None, None

    for folder in os.listdir(model_folder):
        if f'train_{model}' in folder:
            summary_file = os.path.join(model_folder, folder, 'summary.out')


            assert os.path.exists(summary_file), f'{summary_file} not found'
            with open(summary_file, 'r') as f_in:
                for line in f_in.readlines():
                    if line.startswith('Mean'):
                        mean = line.split(':')[1].strip()
                    if line.startswith('StdDev'):
                        stddev = line.split(':')[1].strip()

    return mean, stddev


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
        f_out.write('\\begin{tabular}{c| c c c c}\n')
        f_out.write(
            '\t\\textbf{Model} & \\textbf{Epochs 60 } & \\textbf{Epochs 500} & \\textbf{No Freeze} & \\textbf{Nerve centered } \\\\ \n')
        f_out.write('\t\hline\n')
        f_out.write('\t\hline\n')

        for model, execution in model_parameters.items():
            means = []
            for ex, params in execution.items():
                means.append(params[0])

            formatted = model
            if '_' in model:
                formatted = formatted.replace('_', '\_')

            # if (means[0] >= means[1]) and (means[0] >= means[2]):
            #     f_out.write('\t\\rowcolor{pink} \n')
            #
            if means[3] >= means[2]:
                  f_out.write('\t\\rowcolor{YellowGreen} \n')
            #
            # elif (means[1] >= means[0]) and (means[1] >= means[2]):
            #     f_out.write('\t\\rowcolor{cyan} \n')
            #
            # elif (means[2] >= means[0]) and (means[0] >= means[1]):
            #     f_out.write('\t\\rowcolor{orange} \n')


            f_out.write(f'\t{formatted} & {means[0]} & {means[1]} & {means[2]}& {means[3]}\\\\ \n')
            f_out.write(f'\t\hline\n')

        f_out.write('\end{tabular}\n')
        f_out.write('\label{tab:pythorch_models}\n')
        f_out.write('\end{table}\n')




def compare(args):
    models_parameters = {}

    executions = ['/home/ruben/Escritorio/models_deepcopy/models_e60/models',
    '/home/ruben/Escritorio/models_deepcopy/models_e500/models',
    '/home/ruben/Escritorio/models_deepcopy/models_e500_no_freeze/models',
    '/home/ruben/Escritorio/models_deepcopy/models_e500_cnerve/models']

    for model in ARCHITECTURES:
        models_parameters[model] = {}
        for ex in executions:
            ex_name = ex.split('/')[5]
            models_parameters[model][ex_name] = get_model_parameters(model, model_folder=ex)



    # import pprint
    # pprint.pprint(models_parameters)
    write_latex_table(models_parameters, output_file='templates/comparison.latex')


if __name__ == '__main__':
    compare(sys.argv)
