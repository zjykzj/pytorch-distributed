# -*- coding: utf-8 -*-

"""
@date: 2020/9/14 下午8:36
@file: main.py
@author: zj
@description: 
"""

from datetime import datetime
import argparse
import torch
import torch.nn as nn

from model import ConvNet
from data import build_dataloader

from engine import train_epoch, test_epoch


def load_args():
    parser = argparse.ArgumentParser()
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


def process(gpu, args):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = ConvNet().to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Data loading code
    train_data_loader = build_dataloader(args, train=True)
    test_data_loader = build_dataloader(args, train=False)

    start = datetime.now()
    for epoch in range(args.epochs):
        epoch_start = datetime.now()
        train_epoch(epoch, args, model, device, train_data_loader, optimizer, criterion)
        print("Training one epoch in: " + str(datetime.now() - epoch_start))

    print("Training complete in: " + str(datetime.now() - start))

    start = datetime.now()
    test_epoch(model, device, test_data_loader, criterion)
    print("Training one epoch in: " + str(datetime.now() - start))


def main():
    args = load_args()
    process(0, args)


if __name__ == '__main__':
    main()
