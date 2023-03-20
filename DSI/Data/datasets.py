# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import random
import numpy
import copy
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, Dataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)

class DictDataset:
    def __init__(self,dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x,y = self.dataset[index]
        out_dict = {}
        out_dict["y"] = torch.tensor(y, dtype=torch.long)
        return x,out_dict

    def __len__(self):
        return len(self.dataset)

class MultipleDomainDataset:
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)


class Debug(MultipleDomainDataset):
    def __init__(self, root, test_envs, hparams):
        super().__init__()
        self.input_shape = self.INPUT_SHAPE
        self.num_classes = 2
        self.datasets = []
        for _ in [0, 1, 2]:
            self.datasets.append(
                TensorDataset(
                    torch.randn(16, *self.INPUT_SHAPE),
                    torch.randint(0, self.num_classes, (16,))
                )
            )

class wrapDataset:
    def __init__(self,dataset, name):
        self.dataset = dataset
        self.name = name
        self.resize = transforms.Resize(32)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.name in ["FixedColoredMNIST" ]:
            x,y = self.dataset[idx]
            x = torch.cat((x,torch.zeros_like(x[0:1])),dim = 0)
            x = self.resize(x)
            x = x * 2 - 1
            return x, y
        else:
            return self.dataset[idx]

class MultipleEnvironmentImageFolder(MultipleDomainDataset):
    def __init__(self, root, test_envs, data_augmentation = False, image_size = 224, mean = (1,1,1), std = (1,1,1), **kwargs):
        super().__init__()
        environments = [f.name for f in os.scandir(root) if f.is_dir()]
        environments = sorted(environments)

        transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean, std=std)
        ]) 
        augment_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean, std=std),
        ])

        self.datasets = []
        for i, environment in enumerate(environments):

            if data_augmentation and (i not in test_envs):
                env_transform = augment_transform
            else:
                env_transform = transform

            path = os.path.join(root, environment)
            env_dataset = ImageFolder(path,
                transform=env_transform)

            self.datasets.append(env_dataset)

        self.input_shape = (3, image_size, image_size,)
        self.num_classes = len(self.datasets[-1].classes)

class VLCS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["C", "L", "S", "V"]
    def __init__(self, root, test_envs, **hparams):
        self.dir = os.path.join(root, "VLCS/")
        super().__init__(self.dir, test_envs, **hparams)

class PACS(MultipleEnvironmentImageFolder):
    CHECKPOINT_FREQ = 300
    ENVIRONMENTS = ["A", "C", "P", "S"]
    def __init__(self, root, test_envs, **hparams): 
        self.dir = os.path.join(root, "PACS/")
        hparams["mean"] = (0.485, 0.456, 0.406)
        hparams["std"] = (0.229, 0.224, 0.225)

        super().__init__(self.dir, test_envs, **hparams)