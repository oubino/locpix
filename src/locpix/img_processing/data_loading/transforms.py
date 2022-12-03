"""Module defining transformations to apply to data"""

from torchvision import transforms
import numpy as np


class transform:
    """Wrapper for transforms to allow input
    and label to be transformed together"""

    def __init__(self, transform_list):
        """Args:
        transform_list (list) : List of transforms to be applied"""

        self.transform = transforms.Compose(transform_list)

    def __call__(self, input, label):
        """Args:
        input (numpy array) : Input histogram
        label (numpy array) : Histogram with labels"""

        print("check")
        print(input.shape)
        print(label.shape)

        data = np.concatenate((input, label), axis=0)
        data = self.transform(data)
        input = data[:-1]
        label = data[-1:]

        return input, label
