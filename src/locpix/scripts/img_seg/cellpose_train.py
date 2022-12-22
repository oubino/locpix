#!/usr/bin/env python
"""Cellpose segmentation module

Take in items and train the Cellpose module
"""

import yaml
import os
import argparse
import json
import time

from locpix.preprocessing import datastruc
import numpy as np
import skimage
from locpix.visualise import vis_img
from cellpose import models

def main():

    # Load in config

    parser = argparse.ArgumentParser(
        description="Train cellpose." "If no args are supplied will be run in GUI mode"
    )
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for preprocessing",
    )
    parser.add_argument(
        "-m",
        "--project_metadata",
        action="store_true",
        help="check the metadata for the specified project and" "seek confirmation!",
    )

    args = parser.parse_args()

    # if want to run in headless mode specify all arguments
    #if args.project_directory is None and args.config is None:
    #    config, project_folder = ilastik_output_config.config_gui()

    if args.project_directory is not None and args.config is None:
        parser.error(
            "If want to run in headless mode please supply arguments to"
            "config as well"
        )

    if args.config is not None and args.project_directory is None:
        parser.error(
            "If want to run in headless mode please supply arguments to project"
            "directory as well"
        )

    # headless mode
    if args.project_directory is not None and args.config is not None:
        project_folder = args.project_directory
        # load config
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            #ilastik_output_config.parse_config(config)

    metadata_path = os.path.join(project_folder, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # check metadata
        if args.project_metadata:
            print("".join([f"{key} : {value} \n" for key, value in metadata.items()]))
            check = input("Are you happy with this? (YES)")
            if check != "YES":
                exit()
        # add time ran this script to metadata
        file = os.path.basename(__file__)
        if file not in metadata:
            metadata[file] = time.asctime(time.gmtime(time.time()))
        else:
            print("Overwriting...")
            metadata[file] = time.asctime(time.gmtime(time.time()))
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # load in config
    input_root = os.path.join(project_folder, "annotate/annotated")
    train_files = config["train_files"]
    test_files = config["test_files"]

    # list items
    try:
        files = os.listdir(input_root)
        files = [os.path.splitext(file)[0] for file in files]
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # make necessary folders if not present
    model_folder = os.path.join(project_folder, "cellpose_model")
    if os.path.exists(model_folder):
        raise ValueError(f"Cannot proceed as {model_folder} already exists")
    else:
        os.makedirs(model_folder)

    # check train and test files
    print("Train files")
    print(train_files)
    print("Test files")
    print(test_files)

    print("Converting data to  datasets")

    # convert train files into imgs and masks
    train_files = [os.path.join(input_root, file + '.parquet') for file in train_files]
    test_files = [os.path.join(input_root, file + '.parquet') for file in test_files]

    # train cellpose model
    model = models.CellposeModel(model_type=config["model"])
    imgs, labels = parquet_2_img(train_files)
    test_imgs, test_labels = parquet_2_img(test_files)
    # threshold imgs
    # ?
    model.train(imgs, 
                labels, 
                train_files=train_files, 
                test_data = test_imgs,
                test_labels = test_labels,
                test_files = test_files,
                channels=config['channels'], 
                save_path=model_folder, 
                save_every=config['save_every'], 
                learning_rate=0.001, 
                n_epochs=config['epochs'],
                nimg_per_epoch=config['nimg_per_epoch'],
                min_train_masks=config['min_train_masks'],
                model_name=config['model_name']
    )

def parquet_2_img(files, folder=None, save=False):
    """Convert data from .parquet files to .png files
    for cellpose
    
    Args:
        files (list) : List of files (.parquet) 
        folder (string) : Path to save data to
        save (bool) : Whether to save"""

    imgs = []
    labels = []

    # for file in input
    for datum in files:

        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(datum))

        # convert
        histo, axis_2_chan = item.render_histo()
        label = item.render_seg()

        # transpose to img space
        img = np.transpose(histo, (0, 2, 1))
        label = label.T
        label = label.astype('int32')

        labels.append(label)
        imgs.append(img)

    return imgs, labels


if __name__ == "__main__":
    main()
