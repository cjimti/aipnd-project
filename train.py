#!/usr/bin/env python3
""" train.py
Part 2 of the Udacity AIPND final project submission for Craig Johnston.
train.py train a new network on a specified data set.
"""
__author__ = "Craig Johnston <cjimti@gmail.com>"
__version__ = "0.0.1"
__license__ = "MIT"

import os
import json
import train_args
import train_network


def main():
    """
        Image Classification Network Trainer
    """
    parser = train_args.get_args()
    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s ' + __version__ + ' by ' + __author__)
    cli_args = parser.parse_args()

    # check for data directory
    if not os.path.isdir(cli_args.data_directory):
        print(f'Data directory {cli_args.data_directory} was not found.')
        exit(1)

    # load categories
    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    train_net = train_network.TrainerNet(
        data_dir=cli_args.data_directory,
        output_size=len(cat_to_name),
        hidden_size=cli_args.hidden_units,
        arch=cli_args.arch
    )

    train_net.train(
        epochs=cli_args.epochs,
        learning_rate=cli_args.learning_rate,
        chk_every=50,
        gpu=cli_args.use_gpu,
        test=cli_args.test
    )

    # dataloader = train_utils.load_data(data_dir)

    # check for save directory
    save_dir = cli_args.save_dir
    if not os.path.isdir(save_dir):
        print(f'Directory {save_dir} does not exist. Creating...')
        os.makedirs(save_dir)


if __name__ == '__main__':
    main()
"""
 Ensures main() only gets called if the script is
 executed directly and not as an include.
"""
