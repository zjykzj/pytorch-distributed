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

from single_gpu.model import ConvNet
from single_gpu.data import build_dataloader


def train(gpu, args):
    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    model = ConvNet().to(device)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Data loading code
    data_loader = build_dataloader()

    start = datetime.now()
    total_step = len(data_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    train(0, args)


if __name__ == '__main__':
    main()
