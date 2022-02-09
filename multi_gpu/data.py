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


def build_dataset(root, transform=None, is_train=True):
    data_set = datasets.MNIST(root, train=is_train, transform=transform, download=True)

    return data_set


def build_sampler(data_set, world_size, rank):
    sampler = DistributedSampler(data_set,
                                 num_replicas=world_size,
                                 rank=rank)
    return sampler


def build_dataloader(args, train=True, world_size=1, rank=0):
    transform = build_transform()
    data_set = build_dataset('./data', transform=transform, is_train=train)

    sampler = build_sampler(data_set, world_size, rank)
    data_loader = DataLoader(dataset=data_set,
                             batch_size=args.batch_size if train else args.test_batch_size,
                             num_workers=0,
                             pin_memory=True,
                             sampler=sampler)

    return data_loader
