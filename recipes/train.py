"""Train recipe

Recipe :
    1. Initialise dataset
    2. Initialise dataloader
    3. Train...
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
    with open('recipes/train.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load in config
    batch_size = config['batch_size']
    epochs = config['epochs']

    # folder
    train_folder = os.path.join(config['processed_dir_root'], 'train')
    val_folder = os.path.join(config['processed_dir_root'], 'val')
    test_folder = os.path.join(config['processed_dir_root'], 'test')

    # transform
    # TODO: #4 add in transforms, and ensure specified in config file
    train_transform = None
    val_transform = None
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

    # TODO: #5 configuration for dataloaders

    # initialise dataloaders
    train_loader = L.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = L.DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # initialise model
    model = model_choice(config['model'],
                         train_set)

    # print parameters
    print('\n')
    print('---- Params -----')
    print('\n')
    print('Input features: ', train_set.num_node_features)
    print('Num classes: ', train_set.num_classes)
    print('Batch size: ', batch_size)
    print('Epochs: ', epochs)

    # model summary
    print('\n')
    print('---- Model summary ----')
    print('\n')
    number_nodes = 1000  # this is just for summary, has no bearing on training
    summary(model, input_size=(train_set.num_node_features, number_nodes),
            batch_size=batch_size)

    # train loop
    print('\n')
    print('---- Training... ----')
    print('\n')
    train.train_loop(epochs,
                     model,
                     )
    print('Need checks here to make sure model weights are\
          correct')
    print('\n')
    print('---- Finished training... ----')
    print('\n')

    # save final model
    print('\n')
    print('---- Saving final model... ----')
    print('\n')

                    

    # evaluate perf metrics
    print('Evaluating...')
