import torch
import numpy as np
import random
import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler


def _init_fn(worker_id):
    np.random.seed(int(0))

def get_train_valid_loader(batch_size,
                           path,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - path: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # load the dataset
    train_dataset = datasets.ImageFolder(root=os.path.join(path, "train/"), transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=os.path.join(path, "test/"), transform=train_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(path, "test/"), transform=train_transform)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=shuffle, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    #food101_mean, food101_std = online_mean_and_sd(train_loader)
    #print(f'Mean:{food101_mean}, STD:{food101_std}')

    return (train_loader, valid_loader, test_loader)


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:

        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)