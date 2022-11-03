"""Evaluate recipe

Recipe :
    1. Load in model
    2. Initialise dataloader
    3. Evaluate on test set
"""

import os
import yaml
from heptapods.data_loading import datastruc
import torch_geometric.loader as L
from heptapods.training import train
from heptapods.models import model_choice
from torchsummary import summary
# import torch
# import torch_geometric.transforms as T


if __name__ == "__main__":

    # load yaml
    with open('recipes/evaluate.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load in config
    batch_size = config['batch_size']
    pin_memory = config['pin_memory']
    num_workers = config['num_workers']
    gpu = config['gpu']

    # folder
    test_folder = os.path.join(config['processed_dir_root'], 'test')

    # transform
    # TODO: #4 add in transforms, and ensure specified in config file
    test_transform = None

    # load in test dataset
    test_set = datastruc.SMLMDataset(None,
                                     None,
                                     test_folder,
                                     transform=test_transform,
                                     pre_transform=None,
                                     pre_filter=None,
                                     gpu=gpu)

    # initialise dataloader
    test_loader = L.DataLoader(test_set,
                               batch_size=batch_size,
                               shuffle=False,
                               pin_memory=pin_memory,
                               num_workers=num_workers)

    # load in model
    
    # print parameters
    print('\n')
    print('---- Params -----')
    print('\n')
    print('Input features: ', test_set.num_node_features)
    print('Num classes: ', test_set.num_classes)
    print('Batch size: ', batch_size)

    # model summary
    print('\n')
    print('---- Model summary ----')
    print('\n')
    number_nodes = 1000  # this is just for summary, has no bearing on training
    summary(model, input_size=(test_set.num_node_features, number_nodes),
            batch_size=batch_size)

    # evaluate model
    # TODO: WRITE MODEL.EVAL SOMEWHERE
    # TODO: make ure torch.no_grad() somewhere

    print('\n')
    print('---- Evaluating... ----')
    print('\n')

    
    

  

   
