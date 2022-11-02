"""Process recipe

Recipe :
    1. Create dataset
    2. Process dataset - pre-transform and save to .pt
"""
import random
import os
import yaml
from heptapods.data_loading import datastruc
from functools import partial
# import torch_geometric.transforms as T


def pre_filter(data, inclusion_list=[]):
    """Takes in data item and returns whether
    it should be included in final dataset
    i.e. 1 - yes ; 0 - no

    Args:
        data (torch.geometric.data) : The pytorch
            geometric dataitem part of the dataset
        inclusion_list (list) : List of names
            indicating which data should be included"""

    if data.name in inclusion_list:
        return 1
    else:
        return 0


if __name__ == "__main__":

    # load yaml
    with open('recipes/process.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # split into train/val/test using pre filter
    file_list = os.listdir(config['raw_dir_root'])
    file_list = [file.strip('.parquet') for file in file_list]
    random.shuffle(file_list)
    # split into train/test/val
    train_length = int(len(file_list) * config['train_ratio'])
    test_length = int(len(file_list) * config['test_ratio'])
    val_length = len(file_list) - train_length - test_length
    train_list = file_list[0:train_length]
    val_list = file_list[train_length:train_length + val_length]
    test_list = file_list[train_length + val_length: len(file_list)]

    # folders
    train_folder = os.path.join(config['processed_dir_root'], 'train')
    val_folder = os.path.join(config['processed_dir_root'], 'val')
    test_folder = os.path.join(config['processed_dir_root'], 'test')
    # if output directory not present create it
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)

    # bind arguments to functions
    train_pre_filter = partial(pre_filter, inclusion_list=train_list)
    val_pre_filter = partial(pre_filter, inclusion_list=val_list)
    test_pre_filter = partial(pre_filter, inclusion_list=test_list)

    # TODO: #3 Add in pre-transforms to process @oubino

    # create train dataset
    trainset = datastruc.SMLMDataset(config['hetero'],
                                     config['raw_dir_root'],
                                     train_folder,
                                     transform=None,
                                     pre_transform=None,
                                     # e.g. pre_transform =
                                     # T.RadiusGraph(r=0.0000003,
                                     # max_num_neighbors=1),
                                     pre_filter=train_pre_filter)

    # create val dataset
    valset = datastruc.SMLMDataset(config['hetero'],
                                   config['raw_dir_root'],
                                   val_folder,
                                   transform=None,
                                   pre_transform=None,
                                   pre_filter=val_pre_filter)

    # create test dataset
    testset = datastruc.SMLMDataset(config['hetero'],
                                    config['raw_dir_root'],
                                    test_folder,
                                    transform=None,
                                    pre_transform=None,
                                    pre_filter=test_pre_filter)
