import torch
import os
from torchvision import datasets
from torchvision import transforms

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        path = os.path.split(path)[1]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

def get_train_valid_loader(batch_size,
                           path,
                           shuffle=True,
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
    train_dataset = ImageFolderWithPaths(root=os.path.join(path, "train/"), transform=train_transform)
    test_dataset = ImageFolderWithPaths(root=os.path.join(path, "test/"), transform=train_transform)
    inference_dataset = ImageFolderWithPaths(root=os.path.join(path, "inference/"), transform=train_transform)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=shuffle, batch_size=batch_size, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, batch_size=batch_size, pin_memory=pin_memory,
    )
    inference_loader = torch.utils.data.DataLoader(
        inference_dataset, shuffle=False, batch_size=batch_size, pin_memory=pin_memory,
    )

    #food101_mean, food101_std = online_mean_and_sd(train_loader)
    #print(f'Mean:{food101_mean}, STD:{food101_std}')

    return (train_loader, test_loader, inference_loader)


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