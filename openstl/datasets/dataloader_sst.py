import numpy as np
import torch
import tables
from torch.utils.data import Dataset
import os.path as osp
from openstl.datasets.utils import create_loader
import torch.nn.functional as F
import random


class SSTDataset(Dataset):
    """Sea Surface Temperature Dataset

    Args:
        data_root (str): Path to the .h5 data file.
        training_time (list): List with two elements specifying the min and max index for the data.
        in_steps (int): Number of steps as input.
        out_steps (int): Number of steps as output.
        transform_data (callable, optional): Optional transform to be applied on the data.
        transform_labels (callable, optional): Optional transform to be applied on the labels.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, training_time, in_steps, out_steps,
                 mean=None, std=None,
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        # Load data from .h5 file
        with tables.open_file(data_root, 'r') as file:
            data = np.array(file.root['sst_out'])

        # Extract the data within the range specified by training_time
        self.data = data[training_time[0]:training_time[1]]
        # expand dim
        self.data = np.expand_dims(self.data, axis=1)
        # nanfill to zero
        self.data = np.nan_to_num(self.data, nan=0.0, posinf=0.0, neginf=0.0)

        self.in_steps = in_steps
        self.out_steps = out_steps
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment

        # Calculate the valid indices for slicing the data
        self.valid_idx = np.arange(self.data.shape[0] - (self.in_steps + self.out_steps) + 1)

        # Calculate mean and std
        if mean is not None:
            self.mean = mean
        else:
            self.mean = np.mean(self.data)

        if std is not None:
            self.std = std
        else:
            self.std = np.std(self.data)
        self.data = (self.data - self.mean) / self.std

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, h, w = seqs.shape  # original shape, e.g., [4, 128, 256]
        seqs = F.interpolate(seqs.unsqueeze(0), scale_factor=1 / crop_scale, mode='trilinear').squeeze(0)
        _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(2, ))  # horizontal flip
        return seqs

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, index):
        # Use valid_idx to get the start index for slicing the data
        start_idx = self.valid_idx[index]
        end_idx = start_idx + self.in_steps + self.out_steps

        # Slice the data into input and output sequences
        sample = self.data[start_idx:end_idx]
        data = sample[:self.in_steps]
        labels = sample[self.in_steps:]

        # Apply data and label transforms if they are provided
        if self.transform_data:
            data = self.transform_data(data)
        if self.transform_labels:
            labels = self.transform_labels(labels)

        # Augmentation
        if self.use_augment:
            # Apply your augmentation logic here
            # e.g. data = augment_function(data)
            pass

        return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# # Example usage:
# data_root = './sst_out.h5'  # Specify the path to your .h5 data file
# training_time = [100, 500]  # Specify the range of indices to use for training
# dataset = SSTDataset(data_root=data_root, training_time=training_time, in_steps=10, out_steps=5)
# data, labels = dataset[0]


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              train_time=[0, 13514],
              val_time=[13514, 13880],
              test_time=[13880, 14245],
              in_steps=30,
              out_steps=30,
              level=1,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              **kwargs):

    _dataroot = osp.join(data_root, f'sst', 'sst_out_reduced.h5')
    weather_dataroot = _dataroot if osp.exists(_dataroot) else osp.join(data_root, 'weather')

    train_set = SSTDataset(data_root=weather_dataroot,
                           training_time=train_time,
                           in_steps=in_steps, out_steps=out_steps,
                           use_augment=use_augment)

    vali_set = SSTDataset(weather_dataroot,
                          training_time=val_time,
                          in_steps=in_steps, out_steps=out_steps,
                          use_augment=False,
                          mean=train_set.mean,
                          std=train_set.std)
    test_set = SSTDataset(weather_dataroot,
                          training_time=test_time,
                          in_steps=in_steps, out_steps=out_steps,
                          use_augment=False,
                          mean=train_set.mean,
                          std=train_set.std)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set,  # validation_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test
