# -*- coding: utf-8 -*-

"""
@date: 2022/2/9 下午9:07
@file: engine.py
@author: zj
@description: 
"""

import os
import torch


def train_epoch(epoch, args, model, device, data_loader, optimizer, criterion):
    model.train()
    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('[PID {}]\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                            100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:
                break


def test_epoch(model, device, data_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))

            test_loss += criterion(output, target.to(device)).item()  # sum up batch loss

            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset), 100. * correct / len(data_loader.dataset)))
