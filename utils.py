#-*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from py3nvml.py3nvml import *


__NVML_HANDLE = None
def get_gpu_memory_usage(use_megabytes=False):
    if __NVML_HANDLE == None:
        nvmlInit()
        __NVML_HANDLE = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(__NVML_HANDLE)
    if use_megabytes:
        return {'free': info.free >> 20, 'used': info.used >> 20, 'total': info.total >> 20}
    else:
        return {'free': info.free, 'used': info.used, 'total': info.total}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_accuracy(outputs, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_imagenet_dataloader(train=False, batch_size=8, num_workers=2, pin_memory=True):
    DATASET_ROOT = '/deeplearning_data/ilsvrc2012'
    if train:
        data_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_path = os.path.join(DATASET_ROOT, 'train')
        shuffle = True
    else:
        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        dataset_path = os.path.join(DATASET_ROOT, 'val')
        shuffle = False
    dataset = datasets.ImageFolder(dataset_path, transform=data_transforms)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return data_loader

