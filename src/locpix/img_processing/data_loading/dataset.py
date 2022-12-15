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
        self, input_root, files, input_type, transform
    ):
        """
        Args:

            input_root (string) : Directory containing the input annotated
                SMLM data
            files (list) : List of the files to include from
                the directory in this dataset
            input_type (string) : String representing the data format of the
                input
            transform (pytorch transform) : Transforms to apply to
                the dataset
        """
        self.input_data = [
            os.path.join(input_root, file + input_type) for file in files
        ]
        #self.label_data = [
        #    os.path.join(label_root, file + label_type) for file in files
        #]
        print("input root", input_root)
        #self.input_data, self.label_data = zip(
        #    *sorted(zip(self.input_data, self.label_data))
        #)
        #print("input and label data")
        print(self.input_data)
        #print(self.label_data)
        self.transform = transform

    def preprocess(self, folder):
        """Convert the raw data into data ready for network

        Args:
            folder (string): Path containing folder to save data at
        """

        # join data
        #join_data = zip(self.input_data, self.label_data)

        # make folders to save data at if not already present
        img_folder = os.path.join(folder, 'imgs')
        label_folder = os.path.join(folder, 'labels')
        for folder in [img_folder, label_folder]:
            if os.path.exists(folder):
                raise ValueError(f"Cannot proceed as {folder} already exists")
            else:
                os.makedirs(folder)


        self.img_data = []
        self.label_data = []

        # for file in input
        for datum in self.input_data:

            # load img and label
            #with open(img, "rb") as f:
            #    histo_bad = pkl.load(f)
            
            # check img and label and check img the same
            #histos = []
            #print(type(histo_bad))
            #for key, value in histo_bad.items():
            #    histos.append(value)
            #histo_bad = np.stack(histos)
            #print("histo")
            #print(axis_2_chan)
            #print(histo.shape)
            #print(histo_bad.shape)

            #np.testing.assert_array_equal(histo, histo_bad)

            item = datastruc.item(None, None, None, None)
            item.load_from_parquet(os.path.join(datum))
            #print(item.df)

            # convert
            histo, axis_2_chan = item.render_histo()
            label = item.render_seg()

            # transpose to img space
            img = np.transpose(histo, (0,2,1))
            label = label.T

            #import matplotlib.pyplot as plt
            #plt.imshow(np.log2(img[0,:,:]), origin = 'upper', cmap = 'Greys', alpha=1)
            #plt.imshow(label, origin='upper', cmap='Reds', alpha=.4)
            #plt.show()
            #print("label", label.shape)
            #print('img', img.shape)

            # img & label path
            img_path = os.path.join(img_folder, item.name + '.npy')
            label_path = os.path.join(label_folder, item.name + '.npy')

            # add img and label path to lists
            self.img_data.append(img_path)
            self.label_data.append(label_path)

            # save img and label
            np.save(img_path, img)
            np.save(label_path, label)

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
