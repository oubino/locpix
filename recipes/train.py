"""Training recipe

Recipe :
    1. Create dataset
    2. Create dataloader
    3. Train .. 
"""

import dotenv
import os
import yaml
from heptapods.data_loading import datastruc

if __name__ == "__main__":

    # load yaml
    with open('recipes/train.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # create dataset
    dataset = datastruc.SMLMDataset(config['raw_dir_root'], 
                                    config['processed_dir_root'],
                                    transform=None, pre_transform=None, 
                                    pre_filter=None)

    # create dataloader