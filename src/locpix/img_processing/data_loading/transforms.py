"""Module defining transformations to apply to data"""

from torchvision import transforms
import numpy as np
import torch


class transform:
    """Wrapper for transforms to allow input
    and label to be transformed together"""

    def __init__(self, mean, std, transform_list, dtypeconv=False):
        """Args:
        mean (float) : Mean for normalisation of image
        std (float) : Std for normalisation of image
        transform_list (list) : List of transforms to be applied
        dtypeconv (string) : Whether to convert image"""

        self.mean = mean
        self.std = std
        self.transform = transforms.Compose(transform_list)
        self.dtypeconv = dtypeconv

    def __call__(self, input, label):
        """Args:
        input (numpy array) : Input histogram
        label (numpy array) : Histogram with labels"""

        input = (input - self.mean) / (self.std)
        data = np.stack((input, label), axis=-1)
        data = self.transform(data)
        if self.dtypeconv is True:
            data = data.to(torch.float32)
        input = data[:-1]
        label = data[-1]

        return input, label
