import os, torch, copy
from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.distributed as dist
from torchvision import transforms


def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    if isinstance(data_dir,list) and "ColoredMNIST" in data_dir[0]:
        print("create Colored MNIST dataset")
        dataset = ColoredMINST(
            data_dir,
            shard=dist.get_rank(),
            num_shards=dist.get_world_size(),
            )
    elif isinstance(data_dir,list) and "cdsprites" in data_dir[0]:
        print("create cdsprites dataset")
        D = DspritesDataset(os.path.join(data_dir[0], 'cdsprites'), split=False)
        dataset = CDsprites_Dataset(D, data_dir[0], 32, data_dir[1], 'train')
    else:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=dist.get_rank(),
            num_shards=dist.get_world_size(),
        )

    # ATT updated 0930
    return dataset

    # if deterministic:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
    #     )
    # else:
    #     loader = DataLoader(
    #         dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
    #     )
    # while True:
    #     yield from loader


def _list_image_files_recursively(data_dirs):
    results = []
    if isinstance(data_dirs, list):
        for data_dir in data_dirs:
            results.extend(_list_image_files_recursively(data_dir))
    elif isinstance(data_dirs, str):
        data_dir = data_dirs
        for entry in sorted(bf.listdir(data_dir)):
            full_path = bf.join(data_dir, entry)
            ext = entry.split(".")[-1]
            if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
                results.append(full_path)
            elif bf.isdir(full_path):
                results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


class ColoredMINST(Dataset):
    def __init__(self, image_paths, *args, **kwargs):
        super().__init__()
        self.resolution = 32
        self.dataset = ConcatDataset([
            torch.load(os.path.join(img_path,"data.pt")) for img_path in image_paths
        ])
        self.local_classes = None 
        self.resize = transforms.Resize(32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
    
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        x,y = self.dataset[idx]
        x = torch.cat((x,torch.zeros_like(x[0:1])),dim = 0)
        x = self.resize(x)
        x = x * 2 - 1
        return x.numpy(), out_dict


class CDsprites_Dataset(Dataset):
    """
    Batched dataset for CDsprites. Allows for getting a batch of data given
    a specific domain index.
    """
    def __init__(self, dataset, data_dir, batch_size, num_domains, split):
        # filter out shape 3
        self.indices = np.where(dataset.latents[:, 0] != 3)[0]
        n_splits = [1 / (num_domains + 2)] * (num_domains + 2)

        domain_indices = self.compute_split(n_splits, data_dir)
        self.data_dir = data_dir
        self.latents = dataset.latents
        self.images = dataset.images

        if split=='val':
            self.latents = self.latents[domain_indices[-2]]
            self.images = self.images[domain_indices[-2]]
        elif split=='test':
            self.latents = self.latents[domain_indices[-1]]
            self.images = self.images[domain_indices[-1]]
        elif split=='train':
            self.train_indices = torch.cat(domain_indices[:num_domains], dim = 0)
            self.latents = self.latents[self.train_indices]
            self.images = self.images[self.train_indices]

        self.domain_indices = domain_indices[:num_domains]
        colors = copy.deepcopy(self.latents[:, 0]) - 1
        self.latents = np.concatenate([self.latents, np.expand_dims(colors, -1)], -1)
        self.batch_size = batch_size

        self.color_palattes = self.retrieve_colors(num_domains)
        self.split = split

        self.local_classes = None 

    def reset_batch(self):
        """Reset batch indices for each domain."""
        self.batch_indices, self.batches_left = {}, {}
        for loc, env_idx in enumerate(self.domain_indices):
            self.batch_indices[loc] = torch.split(env_idx[torch.randperm(len(env_idx))], self.batch_size)
            self.batches_left[loc] = len(self.batch_indices[loc])

    def get_batch(self, domain):
        """Return the next batch of the specified domain."""
        batch_index = self.batch_indices[domain][len(self.batch_indices[domain]) - self.batches_left[domain]]
        return torch.stack([self.get_input(i, domain)[0] for i in batch_index]), \
               (torch.Tensor(self.latents[batch_index])[:, 0]-1).long(), None


    def compute_split(self, n_splits, data_dir):
        os.makedirs(os.path.join(data_dir, 'cdsprites'), exist_ok=True)
        path = os.path.join(data_dir, 'cdsprites', f'train_test_val_split.pt')
        if not os.path.exists(path):
            torch.save(torch.tensor(self.indices[torch.randperm(len(self.indices))]), path)
        rand_indices = torch.load(path)
        N = len(rand_indices)

        split_indices = []
        start_idx = 0
        for split in n_splits:
            end_idx = start_idx + int(N * split)
            split_indices.append(rand_indices[start_idx:end_idx])
            start_idx = end_idx
        return split_indices

    def retrieve_colors(self, total_envs=10):
        path = os.path.join(self.data_dir, 'cdsprites', f'colors_{total_envs}.pt')
        return torch.load(path)

    def get_input(self, idx, env):
        image = torch.Tensor(self.images[idx])
        latent = torch.Tensor(self.latents[idx])

        if len(image.shape)>3:
            canvas = torch.zeros_like(image).repeat(1,3,1,1)
            for c, (img, l) in enumerate(zip(image, latent)):
                canvas[c, ...] = self.get_domain_color_palatte(img, l, env)
        else:
            canvas = self.get_domain_color_palatte(image, latent, env)
        return (canvas, latent, None)


    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx])
        latent = torch.Tensor(self.latents[idx])

        if len(image.shape)>3:
            canvas = torch.zeros_like(image).repeat(1,3,1,1)
            for c, (img, l) in enumerate(zip(image, latent)):
                canvas[c, ...] = self.get_color_palatte(img, l)
        else:
            canvas = self.get_color_palatte(image, latent)
        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        canvas = canvas * 2 - 1
        return canvas.numpy(), out_dict
        return (canvas, latent[0].long()-1, latent[0:])

    def get_color_palatte(self, image, latent):
        chosen_color = torch.randint(high=len(self.color_palattes) - 1, size=(1,)).item()
        cc = int(latent[-1].long()) if self.split == 'train' else \
            torch.randint(high=2, size=(1,)).item()
        canvas = self.color_palattes[chosen_color][cc]
        return canvas*image

    def get_domain_color_palatte(self, image, latent, chosen_color):
        cc = int(latent[-1].long())
        canvas = self.color_palattes[chosen_color][cc]
        return canvas*image

    def eval(self, ypreds, ys, metas):
        total = ys.size(0)
        correct = (ypreds == ys).sum().item()
        test_val = [
            {'acc_avg': correct/total},
            f"Accuracy: {correct/total*100:6.2f}%"
        ]
        return test_val


DspritesDataSize = torch.Size([1, 64, 64])
class DspritesDataset(Dataset):
    """2D shapes dataset.
    More info here:
    https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
    """
    def __init__(self, data_root, train=True, train_fract=0.8, split=True, clip=False):
        """
        Args:
            npz_file (string): Path to the npz file.
        """
        filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
        self.npz_file = data_root + '/' + filename
        self.npz_train_file = data_root + '/train_' + filename
        self.npz_test_file = data_root + '/test_' + filename
        if not os.path.isfile(self.npz_file):
            self.download_dataset(self.npz_file)
        if split:
            if not (os.path.isfile(self.npz_train_file) and os.path.isfile(self.npz_test_file)):
                self.split_dataset(data_root, self.npz_file, self.npz_train_file,
                                   self.npz_test_file, train_fract, clip)
            dataset = np.load(self.npz_train_file if train else self.npz_test_file,
                              mmap_mode='r')
        else:
            rdataset = np.load(self.npz_file, encoding='latin1', mmap_mode='r')
            dataset = {'latents': rdataset['latents_values'][:, 1:],  # drop colour
                       'images': rdataset['imgs']}

        self.latents = dataset['latents']
        self.images = dataset['images']

    def download_dataset(self, npz_file):
        from urllib import request
        url = 'https://github.com/deepmind/dsprites-dataset/blob/master/' \
              'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'
        print('Downloading ' + url)
        data = request.urlopen(url)
        with open(npz_file, 'wb') as f:
            f.write(data.read())

    def split_dataset(self, data_root, npz_file, npz_train_file, npz_test_file, train_fract, clip):
        print('Splitting dataset')
        dataset = np.load(npz_file, encoding='latin1', mmap_mode='r')
        latents = dataset['latents_values'][:, 1:]
        images = np.array(dataset['imgs'], dtype='float32')
        images = images.reshape(-1, *DspritesDataSize)
        if clip:
            images = np.clip(images, 1e-6, 1 - 1e-6)

        split_idx = np.int(train_fract * len(latents))
        shuffled_range = np.random.permutation(len(latents))
        train_idx = shuffled_range[range(0, split_idx)]
        test_idx = shuffled_range[range(split_idx, len(latents))]

        np.savez(npz_train_file, images=images[train_idx], latents=latents[train_idx])
        np.savez(npz_test_file, images=images[test_idx], latents=latents[test_idx])

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx]).unsqueeze(0)
        latent = torch.Tensor(self.latents[idx])
        return (image, latent)