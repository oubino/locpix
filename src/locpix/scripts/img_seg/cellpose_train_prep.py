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
import matplotlib.pyplot as plt
#import skimage
from locpix.visualise import vis_img
from cellpose import models
import tifffile

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
    # if args.project_directory is None and args.config is None:
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
            # ilastik_output_config.parse_config(config)

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
    preprocessed_folder = os.path.join(project_folder, "cellpose_train")
    train_folder = os.path.join(preprocessed_folder, "train")
    test_folder = os.path.join(preprocessed_folder, "test")
    folders = [preprocessed_folder,
               train_folder,
               test_folder,
               ]
    for folder in folders:
        if os.path.exists(folder):
            raise ValueError(f"Cannot proceed as {folder} already exists")
        else:
            os.makedirs(folder)

    # check train and test files
    print("Train files")
    print(train_files)
    print("Test files")
    print(test_files)

    # convert files into imgs and masks
    train_files = [os.path.join(input_root, file + ".parquet") for file in train_files]
    test_files = [os.path.join(input_root, file + ".parquet") for file in test_files]
    parquet_2_img(train_files, config["labels"], config["sum_chan"], config['img_threshold'], config['img_interpolate'], train_folder)
    parquet_2_img(test_files, config["labels"], config["sum_chan"], config['img_threshold'], config['img_interpolate'], test_folder)
    
    # threshold imgs
    # ?
    #model.train(
    #    imgs,
    #    labels,
    #    train_files=train_files,
    #    test_data=test_imgs,
    #    test_labels=test_labels,
    #    test_files=test_files,
    #    channels=config["channels"],
    #    save_path=model_folder,
    #    save_every=config["save_every"],
    #    learning_rate=0.001,
    #    n_epochs=config["epochs"],
    #    nimg_per_epoch=config["nimg_per_epoch"],
    #    min_train_masks=config["min_train_masks"],
    #    model_name=config["model_name"],
    #)


def parquet_2_img(files, labels, sum_chan, img_threshold, img_interpolate, folder):
    """Convert data from .parquet files to .png files
    for cellpose and save

    Args:
        files (list) : List of files (.parquet)
        labels (list) : List of channels id by label
            to render in img
        sum_chan (bool) : If True the channels are
            summed
        img_threshold (float) : The threshold for the image
        img_interpolate (string) : How to interpolate the image
        folder (string) : Folder to save data to
        """
    
    # for file in input
    for datum in files:

        item = datastruc.item(None, None, None, None, None)
        item.load_from_parquet(os.path.join(datum))

        # convert
        histo, channel_map, label_map = item.render_histo(labels)
        label = item.render_seg()

        img_info_path = os.path.join(folder, 'img_info.json')
        with open(img_info_path, "w") as outfile:
            json.dump(label_map, outfile)

        # transpose to img space
        label = label.T
        label = label.astype("int32")
        if not sum_chan:
            img = histo[0].T
        elif sum_chan:
            img = histo[0].T + histo[1].T
        else:
            raise ValueError("sum_chan should be true or false")
        img = vis_img.manual_threshold(
            img, img_threshold, how=img_interpolate
        )

        # save
        #img_folder = os.path.join(folder, 'imgs')
        #label_folder = os.path.join(folder, 'labels')
        #folders = [img_folder, label_folder]
        #for folder in folders:
        img_path = os.path.join(folder, item.name + ".tif")
        label_path = os.path.join(folder, item.name + "_masks.tif")

        # save 
        old_img = img
        old_label = label
        tifffile.imwrite(img_path, img)
        tifffile.imwrite(label_path, label)
        #plt.imsave(img_path, img)
        #plt.imsave(label_path, label)
        # load and check
        img = tifffile.imread(img_path)
        label = tifffile.imread(label_path)
        np.testing.assert_array_equal(old_img, img)
        np.testing.assert_array_equal(old_label, label)


if __name__ == "__main__":
    main()
