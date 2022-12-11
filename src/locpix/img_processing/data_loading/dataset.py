"""Dataset module

This module defines the dataset class for SMLM image data"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle as pkl
from locpix.preprocessing import datastruc


class ImgDataset(Dataset):
    """Pytorch dataset for the SMLM data represented as images

    Attributes:
    """

    def __init__(
        self, input_root, label_root, files, input_type, label_type, transform
    ):
        """
        Args:

            input_root (string) : Directory containing the histograms
                of the SMLM data in .npy form
            label_root (string) : Directory containing the labeled
                histograms labels for training
            files (list) : List of the files to include from
                the directory in this dataset
            input_type (string) : String representing the data format of the
                input
            label_type (string) : String representing the data format of the
                label
            transform (pytorch transform) : Transforms to apply to
                the dataset
        """
        self.input_data = [
            os.path.join(input_root, file + input_type) for file in files
        ]
        self.label_data = [
            os.path.join(label_root, file + label_type) for file in files
        ]
        print("input root", input_root)
        self.input_data, self.label_data = zip(
            *sorted(zip(self.input_data, self.label_data))
        )
        print("input and label data")
        print(self.input_data)
        print(self.label_data)
        self.transform = transform

    def preprocess(self, folder):
        """Convert the raw data into data ready for network

        Args:
            folder (string): Path containing folder to save data at
        """

        # join data
        join_data = zip(self.input_data, self.label_data)

        self.img_data = []
        self.label_data = []

        # for file in input
        for img, label in join_data:

            # load img and label
            with open(img, "rb") as f:
                img_bad = pkl.load(f)

            item = datastruc.item(None, None, None, None)
            item.load_from_parquet(os.path.join(label))
            print(item.df)

            # convert
            img, axis_2_chan = item.render_histo()
            label = item.render_seg()

            # check img and label and check img the same
            histos = []
            print(type(img_bad))
            for key, value in img_bad.items():
                histos.append(value)
            img_bad = np.stack(histos)
            print("img")
            print(axis_2_chan)
            print(img.shape)
            print(img_bad.shape)

            np.testing.assert_array_equal(img, img_bad)
            print("label", label.shape)

            # img path

            # label path

            # save img and label

            # add img and label path to lists

    def __getitem__(self, idx):
        """Returns an item from the dataset, according to index idx

        Args:
            idx (int or other) : Index of the data to retrieve"""

        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_path = self.input_data[idx]
        label_path = self.label_data[idx]

        input = np.load(input_path)
        label = np.load(label_path)

        # transform input and label together
        input, label = self.transform(input, label)

        return input, label

    def __len__(self):
        """Length of the dataset

        Args:
            None"""

        return len(self.input_data)
