#!/usr/bin/env python3
""" train.py
Part 2 of the Udacity AIPND final project submission for Craig Johnston.
predict.py Predict flower name from image.
"""
__author__ = "Craig Johnston <cjimti@gmail.com>"
__version__ = "1.0.0"
__license__ = "MIT"

import json
import torch
import predict_args
import warnings

from PIL import Image
from torchvision import models
from torchvision import transforms


def main():
    """
        Image Classification Prediction
    """
    # load the cli args
    parser = predict_args.get_args()
    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s ' + __version__ + ' by ' + __author__)
    cli_args = parser.parse_args()

    # Start with CPU
    device = torch.device("cpu")

    # Requested GPU
    if cli_args.use_gpu:
        device = torch.device("cuda:0")

    # load categories
    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)

    # load model
    chkp_model = load_checkpoint(device, cli_args.checkpoint_file)

    top_prob, top_classes = predict(cli_args.path_to_image, chkp_model, cli_args.top_k)

    label = top_classes[0]
    prob = top_prob[0]

    print(f'Parameters\n---------------------------------')

    print(f'Image  : {cli_args.path_to_image}')
    print(f'Model  : {cli_args.checkpoint_file}')
    print(f'Device : {device}')

    print(f'\nPrediction\n---------------------------------')

    print(f'Flower      : {cat_to_name[label]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob*100:.2f}%')

    print(f'\nTop K\n---------------------------------')

    for i in range(len(top_prob)):
        print(f"{cat_to_name[top_classes[i]]:<25} {top_prob[i]*100:.2f}%")


def predict(image_path, model, topk=5):
    # evaluation mode
    # https://pytorch.org/docs/stable/nn.html#torch.nn.Module.eval
    model.eval()

    # cpu mode
    model.cpu()

    # load image as torch.Tensor
    image = process_image(image_path)

    # Un-squeeze returns a new tensor with a dimension of size one
    # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
    image = image.unsqueeze(0)

    # Disabling gradient calculation
    # (not needed with evaluation mode?)
    with torch.no_grad():
        output = model.forward(image)
        top_prob, top_labels = torch.topk(output, topk)

        # Calculate the exponential
        top_prob = top_prob.exp()

    class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
    mapped_classes = list()

    for label in top_labels.numpy()[0]:
        mapped_classes.append(class_to_idx_inv[label])

    return top_prob.numpy()[0], mapped_classes


def load_checkpoint(device, file='checkpoint.pth'):
    """
    Loads model checkpoint saved by train.py
    """
    # Loading weights for CPU model while trained on GPU
    # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
    model_state = torch.load(file, map_location=lambda storage, loc: storage)

    model = models.__dict__[model_state['arch']](pretrained=True)
    model = model.to(device)

    model.classifier = model_state['classifier']
    model.load_state_dict(model_state['state_dict'])
    model.class_to_idx = model_state['class_to_idx']

    return model


def process_image(image):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    expects_means = [0.485, 0.456, 0.406]
    expects_std = [0.229, 0.224, 0.225]

    pil_image = Image.open(image).convert("RGB")

    # Any reason not to let transforms do all the work here?
    in_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(expects_means, expects_std)])
    pil_image = in_transforms(pil_image)

    return pil_image


if __name__ == '__main__':
    # some models return deprecation warnings
    # https://stackoverflow.com/questions/14463277/how-to-disable-python-warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()


"""
 Ensures main() only gets executed if the script is
 executed directly and not as an include.
"""

