import torch, os, random
from torch.utils.data import random_split, ConcatDataset

from torchvision.datasets import  ImageFolder

def use_shuffle(dataset, dataset_name, domain_index):
    basepath = r"~/Data/shuffled_samples"
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
    return dataset, output


paras = [
    (,"ImageNetR",0),
    (,"ImageNetA",0),
    (,"WaterBird",0),
    (,"OfficeHome",0),
    (,"GTA",0),
]

for root,name,domain in paras:

    dataset = ImageFolder(root)
    use_shuffle(dataset, name, domain)
