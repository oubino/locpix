#!/usr/bin/env python
"""Cellpose segmentation module

Take in items and train the Cellpose module
"""

import yaml

# import tkinter as tk
# from tkinter import filedialog
# import torch
# from torch.utils.data import DataLoader
# from locpix.img_processing.data_loading import dataset
# from locpix.img_processing.training import train
import os

# from torchvision import transforms
# from cellpose import models
# from torchsummary import summary
import argparse
from locpix.scripts.img_seg import img_train_prep
import json
import time
from cellpose import __main__
from locpix.scripts.img_seg import cellpose_eval


def main():

    # Load in config

    parser = argparse.ArgumentParser(description="Train cellpose.")
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
        required=True,
    )
    parser.add_argument(
        "-ct",
        "--config_train",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file for cellpose training",
        required=True,
    )
    parser.add_argument(
        "-ce",
        "--config_eval",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file for cellpose evaluation",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--project_metadata",
        action="store_true",
        help="check the metadata for the specified project and" "seek confirmation!",
    )

    args = parser.parse_args()
    project_folder = args.project_directory
    # load config
    with open(args.config_train, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

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
            print("Overwriting metadata...")
            metadata[file] = time.asctime(time.gmtime(time.time()))
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # for every fold
    folds = len(metadata["train_folds"])

    # cellpose test prep
    # img_train_prep.preprocess_test_files(project_folder, config, metadata)

    # make folder
    cellpose_train_folder = os.path.join(project_folder, "cellpose_train")
    folders = [
        cellpose_train_folder,
    ]
    for folder in folders:
        if os.path.exists(folder):
            raise ValueError(f"Cannot proceed as {folder} already exists")
        else:
            os.makedirs(folder)

    print("------ Training --------")

    for fold in range(folds):

        print(f"----- Fold {fold} -------")

        # cellpose train prep
        img_train_prep.preprocess_train_files(
            project_folder,
            config,
            metadata,
            fold,
            "cellpose",
        )

        # train cellpose
        train_folder = os.path.abspath(
            os.path.join(project_folder, "train_files/cellpose/train")
        )
        test_folder = os.path.abspath(
            os.path.join(project_folder, "train_files/cellpose/val")
        )
        model_save_path = os.path.abspath(
            os.path.join(project_folder, "cellpose_train/")
        )
        model = config["model"]
        lr = config["learning_rate"]
        wd = config["weight_decay"]
        epochs = config["epochs"]

        if config["use_gpu"]:
            __main__.main(
                [
                    "--train",
                    f"--dir={train_folder}",
                    f"--test_dir={test_folder}",
                    f"--pretrained_model={model}",
                    "--chan=0",
                    "--chan2=0",
                    f"--learning_rate={lr}",
                    f"--weight_decay={wd}",
                    f"--n_epochs={epochs}",
                    "--min_train_masks=1",
                    "--verbose",
                    f"--fold={fold}",
                    f"--model_save_path={model_save_path}",
                    "--use_gpu",
                ]
            )
        else:
            __main__.main(
                [
                    "--train",
                    f"--dir={train_folder}",
                    f"--test_dir={test_folder}",
                    f"--pretrained_model={model}",
                    "--chan=0",
                    "--chan2=0",
                    f"--learning_rate={lr}",
                    f"--weight_decay={wd}",
                    f"--n_epochs={epochs}",
                    "--min_train_masks=1",
                    "--verbose",
                    f"--fold={fold}",
                    f"--model_save_path={model_save_path}",
                ]
            )

        # clean up
        img_train_prep.clean_up(project_folder, "cellpose")

    print("------ Outputting for evaluation -------- ")

    for fold in range(folds):

        # load model
        model = os.listdir(
            os.path.join(project_folder, f"cellpose_train/models/{fold}")
        )[0]
        model = os.path.abspath(
            os.path.join(project_folder, f"cellpose_train/models/{fold}/{model}")
        )

        output_folder = os.path.join(cellpose_train_folder, f"{fold}")

        if os.path.exists(output_folder):
            raise ValueError(f"Cannot proceed as {output_folder} already exists")
        else:
            os.makedirs(output_folder)

        # run cellpose_eval
        with open(args.config_eval, "r") as ymlfile:
            config_eval = yaml.safe_load(ymlfile)
        cellpose_eval.main(
            (
                [
                    f"--project_directory={project_folder}",
                    f"--config={config_eval}",
                    f"--output_folder=cellpose_train/{fold}",
                    f"--user_model={model}",
                ]
            )
        )

    # save train yaml file
    yaml_save_loc = os.path.join(project_folder, "cellpose_train.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)

    # save eval yaml file
    yaml_save_loc = os.path.join(project_folder, "cellpose_train_eval.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config_eval, outfile)


if __name__ == "__main__":
    main()
