"""Process recipe

Recipe :
    1. Create dataset
    2. Process dataset - pre-transform and save to .pt
"""

import dotenv
import os
import yaml
from heptapods.data_loading import datastruc
import torch_geometric.transforms as T


if __name__ == "__main__":

    # load yaml
    with open('recipes/process.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # create dataset
    dataset = datastruc.SMLMDataset(config['hetero'],
                                    config['raw_dir_root'], 
                                    config['processed_dir_root'],
                                    transform=None, pre_transform=None,# T.RadiusGraph(r=0.0000003, max_num_neighbors=1), 
                                    pre_filter=None)
    