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
from torch.utils.data.distributed import DistributedSampler


def build_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    return transform


def build_dataset(root, transform=None):
    data_set = datasets.MNIST(root, train=True, transform=transform, download=True)

    return data_set


def build_sampler(data_set, world_size, rank):
    sampler = DistributedSampler(data_set,
                                 num_replicas=world_size,
                                 rank=rank)
    return sampler


def build_dataloader(world_size, rank):
    transform = build_transform()
    data_set = build_dataset('./data', transform=transform)

    sampler = build_sampler(data_set, world_size, rank)
    data_loader = DataLoader(dataset=data_set,
                             batch_size=64,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True,
                             sampler=sampler)

    return data_loader
