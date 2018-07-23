#!/usr/bin/env python3
""" train_args.py
Part 2 of the Udacity AIPND final project submission for Craig Johnston.
predict_args.py contains the command line argument definitions for predict.py
"""

import argparse


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
        description="Image prediction.",
        usage="python ./predict.py /path/to/image.jpg checkpoint.pth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('path_to_image',
                        help='Path to image file.',
                        action="store")

    parser.add_argument('checkpoint_file',
                        help='Path to checkpoint file.',
                        action="store")

    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str,
                        help='Directory to save training checkpoint file',
                        )

    parser.add_argument('--top_k',
                        action="store",
                        default=5,
                        dest='top_k',
                        type=int,
                        help='Return top KK most likely classes.',
                        )

    parser.add_argument('--category_names',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str,
                        help='Path to file containing the categories.',
                        )

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False,
                        help='Use GPU')

    parser.parse_args()
    return parser


def main():
    """
        Main Function
    """
    print(f'Command line argument utility for predict.py.\nTry "python train.py -h".')


if __name__ == '__main__':
    main()
"""
 main() is called if script is executed on it's own.
"""
