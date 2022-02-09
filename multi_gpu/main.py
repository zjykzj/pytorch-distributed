# -*- coding: utf-8 -*-

"""
@date: 2020/9/14 下午8:36
@file: main.py
@author: zj
@description:
refer to
1. [PyTorch Distributed Training](https://leimao.github.io/blog/PyTorch-Distributed-Training/)
2. [PyTorch Distributed Evaluation](https://leimao.github.io/blog/PyTorch-Distributed-Evaluation/)
"""

from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist

from model import ConvNet
from data import build_dataloader
from engine import train_epoch, test_epoch


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int,
                        help='Local rank. Necessary for using the torch.distributed.launch utility. (default: 0)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('-e', '--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass (default: true)')

    return parser.parse_args()


def process(args):
    dist.init_process_group(backend='nccl', rank=args.local_rank)
    torch.manual_seed(args.local_rank)
    assert args.local_rank == dist.get_rank()
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device(local_rank)
    print(local_rank, device, world_size)
    model = ConvNet().to(device)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model)

    # define loss function (criterion) and optimizer
    # criterion = nn.CrossEntropyLoss().cuda(gpu)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Data loading code
    train_data_loader = build_dataloader(args, train=True, world_size=world_size, rank=local_rank)
    test_data_loader = build_dataloader(args, train=False, world_size=world_size, rank=local_rank)

    start = datetime.now()
    for epoch in range(args.epochs):
        epoch_start = datetime.now()
        train_epoch(epoch, args, model, device, train_data_loader, optimizer, criterion)
        if local_rank == 0:
            print("Training one epoch in: " + str(datetime.now() - epoch_start))

    if local_rank == 0:
        print("Training complete in: " + str(datetime.now() - start))

    start = datetime.now()
    test_epoch(model, device, test_data_loader, criterion)
    if local_rank == 0:
        print("Training one epoch in: " + str(datetime.now() - start))


def main():
    args = load_args()
    process(args)


if __name__ == '__main__':
    main()
