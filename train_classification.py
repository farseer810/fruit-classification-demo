#-*- coding: utf-8 -*-
import torch
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import AverageMeter, calculate_accuracy


def get_classes(dataset_path):
    dataset = datasets.ImageFolder(dataset_path)
    return dataset.classes
    

def get_dataloader(dataset_path, train=False, batch_size=8, num_workers=2, pin_memory=True):
    if train:
        data_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        shuffle = True
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        shuffle = False
    dataset = datasets.ImageFolder(dataset_path, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return dataloader


def train(model, dataloader, criterion, optimizer, device=None):
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    time_counter = 0
    model.train()
    with tqdm(dataloader) as bar:
        bar.set_postfix(loss='{:.07}'.format(losses.val), top1='{:05d}'.format(int(top1.avg * 1000)), top5='{:05d}'.format(int(top5.avg * 1000)))
        for (inputs, targets) in bar:
            now = time.time()
            batch_size = targets.size(0)

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            losses.update(loss.item(), batch_size)

            acc1, acc5 = calculate_accuracy(outputs, targets, topk=(1, 5)) 
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_counter += time.time() - now
            if time_counter > 0.2:
                bar.set_postfix(loss='{:.07}'.format(losses.val), top1='{:05d}'.format(int(top1.avg * 1000)), top5='{:05d}'.format(int(top5.avg * 1000)))
                time_counter %= 0.2


def evaluate(model, dataloader, criterion, device=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    time_counter = 0
    with torch.no_grad():
        with tqdm(dataloader) as bar:
            bar.set_postfix(loss='{:.07}'.format(losses.val), top1='{:05d}'.format(int(top1.avg * 1000)), top5='{:05d}'.format(int(top5.avg * 1000)))
            for (inputs, targets) in bar:
                now = time.time()
                batch_size = targets.size(0)

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                losses.update(loss.item(), batch_size)

                acc1, acc5 = calculate_accuracy(outputs, targets, topk=(1, 5)) 
                top1.update(acc1[0], batch_size)
                top5.update(acc5[0], batch_size)
                time_counter += time.time() - now
                if time_counter > 0.2:
                    bar.set_postfix(loss='{:.07}'.format(losses.val), top1='{:05d}'.format(int(top1.avg * 1000)), top5='{:05d}'.format(int(top5.avg * 1000)))
                    time_counter %= 0.2

