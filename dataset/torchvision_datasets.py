import torchvision
import os
import sys
import matplotlib.pyplot as plt
import logging

images_folder = os.path.join(os.path.expanduser("~"), ".pytorch")


def get_torchvision_cifar10_dataset(isTrain=True):
    _dataset = torchvision.datasets.CIFAR10(root=images_folder, download=True, train=isTrain)
    logging.info(f"CIFAR10 dataset has {len(_dataset)} records")
    return _dataset
