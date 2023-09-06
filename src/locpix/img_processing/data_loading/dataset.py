"""Dataset module

This module defines the dataset class for SMLM image data"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import tifffile
from . import transforms
import torchvision.transforms as T


class ImgDataset(Dataset):
    """Pytorch dataset for the SMLM data represented as images

    Attributes:
    """

    def __init__(self, input_root, files, transform, train=False, mean=0, std=1):
        """
        Args:

            input_root (string) : Directory containing the SMLM data and masks
            files (list) : List of the files to include from
                the directory in this dataset
            transform (dictionary) : Transforms to apply to
                the dataset
        """
        self.input_data = [os.path.join(input_root, file + ".tif") for file in files]
        self.label_data = [
            os.path.join(input_root, file + "_masks.tif") for file in files
        ]
        self.input_data, self.label_data = zip(
            *sorted(zip(self.input_data, self.label_data))
        )

        if train:
            # calculate mean and standard deviation
            for index, file in enumerate(self.input_data):
                image = tifffile.imread(file)
                if index == 0:
                    output_image = image
                else:
                    output_image = np.concatenate((output_image, image))
            self.mean = np.mean(output_image, axis=(0, 1))
            self.std = np.std(output_image, axis=(0, 1))
        else:
            self.mean = mean
            self.std = std

        # define transforms
        output_transforms = []

        # to tensor
        output_transforms.append(T.ToTensor())

        # random rotation
        if "rotation" in transform.keys():
            output_transforms.append(T.RandomRotation(transform["rotation"]))

        # random horizontal flip
        if "h_flip" in transform.keys():
            output_transforms.append(T.RandomHorizontalFlip())

        # random vertical flip
        if "v_flip" in transform.keys():
            output_transforms.append(T.RandomVerticalFlip())

        # random erasing
        if "erasing" in transform.keys():
            output_transforms.append(T.RandomErasing())

        # random perspective
        if "perspective" in transform.keys():
            output_transforms.append(T.RandomPerspective(transform["perspective"]))

        # convert to float32
        if "dtypeconv" in transform.keys():
            self.transform = transforms.transform(
                self.mean, self.std, output_transforms, dtypeconv=True
            )
        else:
            self.transform = transforms.transform(
                self.mean, self.std, output_transforms, dtypeconv=False
            )

    def __getitem__(self, idx):
        """Returns an item from the dataset, according to index idx

        Args:
            idx (int or other) : Index of the data to retrieve"""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_path = self.input_data[idx]
        label_path = self.label_data[idx]

        input = tifffile.imread(input_path)
        label = tifffile.imread(label_path)

        input, label = self.transform(input, label)

        return input, label

    def __len__(self):
        """Length of the dataset

        Args:
            None"""

        return len(self.input_data)
