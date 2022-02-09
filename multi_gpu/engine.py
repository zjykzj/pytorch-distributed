# -*- coding: utf-8 -*-

"""
@date: 2022/2/9 下午9:07
@file: engine.py
@author: zj
@description: 
"""

import os
import torch
import torchmetrics
import torch.distributed as dist


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
            if args.local_rank == 0:
                print('[PID {}]\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                                100. * batch_idx / len(data_loader), loss.item()))
            if args.dry_run:
                break


def test_epoch(model, device, data_loader, criterion):
    model.eval()
    # initialize model
    metric = torchmetrics.Accuracy().to(device)

    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            pred = output.max(1)[1]  # get the index of the max log-probability

            # print(pred)
            # print(target)
            acc = metric(pred.to(device), target.to(device))

    # metric on all batches and all accelerators using custom accumulation
    # accuracy is same across both accelerators
    acc = metric.compute()
    # if dist.get_rank() == 0:
    print(f"Accuracy on all data: {acc}, accelerator rank: {dist.get_rank()}")

    # Reseting internal state such that metric ready for new data
    metric.reset()
