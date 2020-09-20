# -*- coding: utf-8 -*-

"""
@date: 2020/9/14 下午8:36
@file: data.py
@author: zj
@description: 
"""

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def build_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform


def build_dataset(root, transform=None):
    data_set = datasets.MNIST(root, train=True, transform=transform, download=True)

    return data_set


def build_dataloader():
    transform = build_transform()
    data_set = build_dataset('./data', transform=transform)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=64,
                             shuffle=True,
                             num_workers=0,
                             pin_memory=True)

    return data_loader
