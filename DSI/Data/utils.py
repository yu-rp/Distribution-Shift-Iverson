import torch, os, random
from torch.utils.data import random_split, ConcatDataset

from .datasets import MultipleEnvironmentMNIST

def random_train_test_split(dataset,train_ratio):
    assert train_ratio <= 1 and train_ratio >=0 
    train_len  = int(len(dataset)*train_ratio)
    test_len = len(dataset) - train_len

    return random_split(dataset,[train_len,test_len])

def leave_one_out(datasets, test_indices, train_indices = None):
    # import pdb
    # pdb.set_trace()
    tests, trains = [],[]
    for i in range(len(datasets)):
        if i in test_indices:
            tests.append(datasets[i])
        elif train_indices is None or i in train_indices:
            trains.append(datasets[i])
    return ConcatDataset(trains) if len(trains) != 0 else None, ConcatDataset(tests) if len(tests) != 0 else None

def use_shuffle(dataset, dataset_name, domain_index):
    if "samples" in vars(dataset).keys():
        basepath = r"~/Repos/DiffusionAttributionCompelete_mulgpu/Data/shuffled_samples"
        filename = f"{dataset_name}_{domain_index}.pt"
        fullpath = os.path.join(basepath, filename)
        if os.path.exists(fullpath):
            samples = torch.load(fullpath,map_location="cpu")
            output = f"shuffled index exists at {fullpath} and loaded"
        else:
            samples = dataset.imgs
            random.shuffle(samples)
            torch.save(samples, fullpath)
            output = f"shuffled index does not exist, is created and saved at {fullpath}"
        dataset.samples = samples
        dataset.imgs = samples
    else:
        assert dataset.fixed
        output = "Fixed Dataset, no shuffle"
    return dataset, output