#!/usr/bin/env python3
""" train_args.py
Part 2 of the Udacity AIPND final project submission for Craig Johnston.
train_args.py contains the command line argument definitions for train.py
"""

import argparse

import train_network


def get_args():
    """
    Get argument parser for train cli.

    Command line argument examples:
    - Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    - Choose architecture: python train.py data_dir --arch "vgg13"
    - Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
    - Use GPU for training: python train.py data_dir --gpu

    For argparse examples see https://pymotw.com/3/argparse
    Returns an argparse parser.
    """

    parser = argparse.ArgumentParser(
        description="Train and save an image classification",
        usage="python ./train.py ./flowers/train --gpu --learning_rate 0.001 --hidden_units 3136 --epochs 5",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('data_directory', action="store")

    parser.add_argument('--save_dir',
                        action="store",
                        default="./",
                        dest='save_dir',
                        type=str,
                        help='Directory to save training checkpoint file',
                        )

    parser.add_argument('--save_name',
                        action="store",
                        default="checkpoint",
                        dest='save_name',
                        type=str,
                        help='Checkpoint filename.',
                        )

    parser.add_argument('--categories_json',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str,
                        help='Path to file containing the categories.',
                        )

    parser.add_argument('--arch',
                        action="store",
                        default="vgg16",
                        dest='arch',
                        type=str,
                        help='Supported architectures: ' + ", ".join(train_network.supported_arch),
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False,
                        help='Use GPU')

    parser.add_argument('-t',
                        action="store_true",
                        dest="test",
                        default=False,
                        help='Testing (process one image)')

    hp = parser.add_argument_group('hyperparameters')

    hp.add_argument('--learning_rate',
                    action="store",
                    default=0.001,
                    type=float,
                    help='Learning rate')

    hp.add_argument('--hidden_units', '-hu',
                    action="store",
                    dest="hidden_units",
                    default=[3136,784],
                    type=int,
                    nargs='+',
                    help='Hidden layer units')

    hp.add_argument('--epochs',
                    action="store",
                    dest="epochs",
                    default=1,
                    type=int,
                    help='Epochs')

    parser.parse_args()
    return parser


def main():
    """
        Main Function
    """
    print(f'Command line argument utility for train.py.\nTry "python train.py -h".')


if __name__ == '__main__':
    main()
"""
 main() is called if script is executed on it's own.
"""
