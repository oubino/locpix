"""Train recipe

Recipe :
    1. Initialise dataset
    2. Initialise dataloader
    3. Train...
"""

import dotenv
import os
import yaml
from heptapods.data_loading import datastruc
import torch_geometric.transforms as T
import torch_geometric.loader as L
import torch


if __name__ == "__main__":

    # load yaml
    with open('recipes/train.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load in train dataset
    train_set = datastruc.SMLMDataset(None,
                                    None, 
                                    config['processed_dir_root'],
                                    transform=None, pre_transform=None,
                                    pre_filter=None)
    

    # load in val dataset
    val_set = 

    # load in test dataset
    test_set = 


    # initialise dataloaders
    train_loader = L.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = L.DataLoader(test_set, batch_size=1, shuffle=False)
    val_loader = L.DataLoader(val_set, batch_size=1, shuffle=False)

    # train loop

    # evaluate perf metrics

    