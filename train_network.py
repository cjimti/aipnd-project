#!/usr/bin/env python3
""" train_args.py
TrainerNet Class

Part 2 of the Udacity AIPND final project submission for Craig Johnston.
train_network.py contains TrainerNet class for configuring and training a neural network model.
"""

import torch

from collections import OrderedDict
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

supported_arch = [
    'vgg11',
    'vgg13',
    'vgg16',
    'vgg19',
    'densenet121',
    'densenet169',
    'densenet161',
    'densenet201'
]

class TrainerNet:
    """
    TrainerNet

    Class for configuring and training a neural network model.
    """
    def __init__(self, data_dir, output_size, hidden_size, arch='vgg16'):
        """
        Construct a TrainerNet

        :param data_dir: string, image directories
        :param output_size: int, number of outputs (match number of categories)
        :param hidden_size: int, size of hidden layer
        :param arch: string, pre-trained arch type (default vgg16)
        """

        print(f'Network Setup\n----------------------------')
        print(f'Net Arch:       {arch}')
        print(f'Hidden Size:    {hidden_size}')
        print(f'Output Size:    {output_size}')
        print(f'Data Directory: {data_dir}')

        self._arch = arch
        self._data_loader, self._class_to_idx = _get_data_loader(data_dir)
        self._model = self._get_net(output_size, hidden_size, self._arch)  # type: nn.Module

    def __str__(self):
        """
        String representation of a TrainerNet object.
        :return: string
        """
        return f'TrainerNet: Pre-trained {self.arch} network.'

    def get_model(self):
        """
        Get the configured nn.Module
        :return: nn.Module
        """
        return self._model

    def train(self, epochs=5, learning_rate=0.001, chk_every=50, dir="./", name="checkpoint", gpu=True, test=False):
        """
        Train the model
        :return: nn.Module
        """
        device = torch.device("cpu")

        if gpu and not torch.cuda.is_available():
            print(f'No GPU. Cuda is not available.')
            exit(1)

        if gpu and torch.cuda.is_available():
            device = torch.device("cuda:0")

        # Start clean by setting gradients of all parameters to zero.
        self._model.zero_grad()

        # The negative log likelihood loss as criterion.
        criterion = nn.NLLLoss()

        # Adam: A Method for Stochastic Optimization
        # https://arxiv.org/abs/1412.6980
        optimizer = optim.Adam(self._model.classifier.parameters(), lr=learning_rate)

        # Move model to perferred device.
        self._model = self._model.to(device)

        data_set_len = len(self._data_loader.batch_sampler)

        print(f'\nBegin Training\n----------------------------')
        print(f'Using the {device} device to train.')
        print(f'Training on {data_set_len} images.')
        print(f'Displaying average loss and accuracy for epoch every {chk_every} images.')
        print(f'Epochs: {epochs} Learning Rate: {learning_rate}')

        for e in range(epochs):
            e_loss = 0
            prev_chk = 0
            total = 0
            correct = 0
            print(f'\nEpoch {e+1} of {epochs}\n----------------------------')
            for ii, (images, labels) in enumerate(self._data_loader):
                # Move images and labeles perferred device
                # if they are not already there
                images = images.to(device)
                labels = labels.to(device)

                # Set gradients of all parameters to zero.
                optimizer.zero_grad()

                # Propigate forward and backward
                outputs = self._model.forward(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Keep a running total of loss for
                # this epoch
                e_loss += loss.item()

                # Accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Keep a running total of loss for
                # this epoch
                itr = (ii + 1)
                if itr % chk_every == 0 or test is True:
                    avg_loss = f'avg. loss: {e_loss/itr:.4f}'
                    acc = f'accuracy: {(correct/total) * 100:.2f}%'
                    print(f'  Images {prev_chk:03} to {itr:03}: {avg_loss}, {acc}.')
                    prev_chk = (ii + 1)

                if test is True:
                    print("END: Test mode runs one image.")
                    break

        print('Done... Saving')

        self._model.class_to_idx = self._class_to_idx
        model_state = {
            'epoch': epochs,
            'state_dict': self._model.state_dict(),
            'optimizer_dict': optimizer.state_dict(),
            'classifier': self._classifier,
            'class_to_idx': self._model.class_to_idx,
            'arch': self._arch
        }

        torch.save(model_state, f'{dir}/{name}.pth')

    def _get_net(self, output_size, hidden_size, arch='vgg16'):
        """
        Returns a vgg or pre-trained network model.

        """
        if not arch.startswith("vgg") and not arch.startswith("densenet"):
            print("Only supporting VGG and DenseNet")
            exit(1)

        nn_model = models.__dict__[arch](pretrained=True)

        densenet_input = {
            'densenet121': 1024,
            'densenet169': 1664,
            'densenet161': 2208,
            'densenet201': 1920
        }

        input_size = 0

        # Input size from current classifier if VGG
        if arch.startswith("vgg"):
            input_size = nn_model.classifier[0].in_features

        if arch.startswith("densenet"):
            input_size = densenet_input[arch]

        # Prevent back propagation on pre-trained parameters
        for param in nn_model.parameters():
            param.requires_grad = False

        # Create nn.Module with Sequential using an OrderedDict
        # See https://pytorch.org/docs/stable/nn.html#torch.nn.Sequential
        od = OrderedDict()

        hidden_size.insert(0, input_size)

        for i in range(len(hidden_size) - 1):
            od['fc' + str(i)] = nn.Linear(hidden_size[i], hidden_size[i + 1])
            od['relu' + str(i)] = nn.ReLU()
            od['dropout' + str(i)] = nn.Dropout(p=0.15)

        od['output'] = nn.Linear(hidden_size[i + 1], output_size)
        od['softmax'] = nn.LogSoftmax(dim=1)

        self._classifier = nn.Sequential(od)

        # Replace classifier
        nn_model.classifier = self._classifier

        return nn_model


def _get_data_loader(data_dir, batch_size=32):
    """
    Load and transform image data.
    """

    # Pre-trained network expectations
    expected_means = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    max_image_size = 224

    training_transformer = transforms.Compose([transforms.RandomHorizontalFlip(p=0.25),
                                               transforms.RandomRotation(25),
                                               transforms.RandomGrayscale(p=0.02),
                                               transforms.RandomResizedCrop(max_image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize(expected_means, expected_std)])

    image_dataset = datasets.ImageFolder(data_dir, transform=training_transformer)

    return torch.utils.data.DataLoader(image_dataset, batch_size=batch_size), image_dataset.class_to_idx


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
