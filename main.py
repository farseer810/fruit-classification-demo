#!/usr/bin/env python
#-*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from train_classification import train, get_classes, get_dataloader, evaluate


def main():
    DATASET_ROOT = '/deeplearning_data/fruits'
    num_epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader(os.path.join(DATASET_ROOT, 'train'), train=True, batch_size=16)
    val_loader = get_dataloader(os.path.join(DATASET_ROOT, 'val'), train=False)
    model = models.densenet201(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        train(model, train_loader, criterion, optimizer, device)
        evaluate(model, val_loader, criterion, device)


if __name__ == '__main__':
    main()
