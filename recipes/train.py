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

    # folder
    train_folder = os.path.join(config['processed_dir_root'], 'train')
    val_folder = os.path.join(config['processed_dir_root'], 'val')
    test_folder = os.path.join(config['processed_dir_root'], 'test')

    # transform
    train_transform = None
    val_transform = None
    test_transform = None

    # Transforms that appear to be good
    # normalize rotation
    # normalizes scale
    # random jitter
    # random flip
    # random rotate
    # random shear
    # normalize features
    # knngraph
    # radius grpah
    # gdc
    # gcnnorm
    # feature propagation
    # e.g. T.compose([T.ToUndirected(), T.AddSelfLoops()])

    # load in train dataset
    train_set = datastruc.SMLMDataset(None,
                                      None,
                                      train_folder,
                                      transform=train_transform,
                                      pre_transform=None,
                                      pre_filter=None)
    

    # load in val dataset
    val_set = datastruc.SMLMDataset(None,
                                    None,
                                    val_folder,
                                    transform=val_transform,
                                    pre_transform=None,
                                    pre_filter=None)

    # load in test dataset
    test_set = datastruc.SMLMDataset(None,
                                     None,
                                     test_folder,
                                     transform=test_transform,
                                     pre_transform=None,
                                     pre_filter=None)


    # initialise dataloaders
    train_loader = L.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = L.DataLoader(test_set, batch_size=1, shuffle=False)
    val_loader = L.DataLoader(val_set, batch_size=1, shuffle=False)

    # train loop
    print('Training...')

    

    # evaluate perf metrics

    