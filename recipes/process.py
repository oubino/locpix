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

    def pre_filter(data, inclusion_list=None):
        """Takes in data item and returns whether
        it should be included in final dataset
        i.e. 1 - yes ; 0 - no
        
        Args:
            data (torch.geometric.data) : The pytorch
                geometric dataitem part of the dataset
            name (string) : Name of the dataitem
            inclusion_list (list) : List of names
                indicating which data should be included"""
        
        if data.name in inclusion_list:
            return 1
        else:
            return 0 

    # split into train/val/test using pre filter


    # create train dataset
    dataset = datastruc.SMLMDataset(config['hetero'],
                                    config['raw_dir_root'], 
                                    config['processed_dir_root'],
                                    transform=None, pre_transform=None,# T.RadiusGraph(r=0.0000003, max_num_neighbors=1), 
                                    pre_filter=None)

    # create val dataset

    # create test dataset
