#!/usr/bin/env python
"""Train prep module

Split the dataset into train and test
Save these splits and also perform k fold split
Save all results in metadata and
"""

import yaml
import os
import argparse
import json
import time
from sklearn.model_selection import KFold


def main():

    parser = argparse.ArgumentParser(description="Train prep")
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for train prep",
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
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    input_folder = os.path.join(project_folder, "annotate/annotated")
    try:
        os.listdir(input_folder)
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # split train/test
    train_files = config["train_files"]
    test_files = config["test_files"]

    # check files
    if not set(train_files).isdisjoint(test_files):
        raise ValueError("Train files and test files shared files!!")
    if len(set(train_files)) != len(train_files):
        raise ValueError("Train files contains duplicates")
    if len(set(test_files)) != len(test_files):
        raise ValueError("Test files contains duplicates")

    # create train and test folder
    train_folder = os.path.join(project_folder, "train_files")
    test_folder = os.path.join(project_folder, "test_files")
    folders = [
        train_folder,
        test_folder,
    ]
    for folder in folders:
        if os.path.exists(folder):
            raise ValueError(f"Cannot proceed as {folder} already exists")
        else:
            os.makedirs(folder)

    # split train into folds
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True)

    train_folds = []
    val_folds = []

    for (train_index, val_index) in kf.split(train_files):

        train_fold = []
        val_fold = []

        for index in train_index:
            train_fold.append(train_files[index])

        for index in val_index:
            val_fold.append(train_files[index])

        train_folds.append(train_fold)
        val_folds.append(val_fold)

    # check files
    for fold in range(folds):
        # check train val test files
        train_files = train_folds[fold]
        val_files = val_folds[fold]
        # check files
        if not set(train_files).isdisjoint(test_files):
            raise ValueError("Train files and test files shared files!!")
        if not set(train_files).isdisjoint(val_files):
            raise ValueError("Train files and val files shared files!!")
        if not set(val_files).isdisjoint(test_files):
            raise ValueError("Val files and test files shared files!!")
        if len(set(train_files)) != len(train_files):
            raise ValueError("Train files contains duplicates")
        if len(set(val_files)) != len(val_files):
            raise ValueError("Val files contains duplicates")
        if len(set(test_files)) != len(test_files):
            raise ValueError("Test files contains duplicates")
        print("Fold", fold)
        print("Train files")
        print(train_files)
        print("Val files")
        print(val_files)
    print("Test files")
    print(test_files)

    # load and add to metadata
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
        # add train val test split
        metadata["train_files"] = train_files + val_files
        metadata["test_files"] = test_files
        metadata["train_folds"] = train_folds
        metadata["val_folds"] = val_folds
        # add time ran this script to metadata
        file = os.path.basename(__file__)
        if file not in metadata:
            metadata[file] = time.asctime(time.gmtime(time.time()))
        else:
            print("Overwriting metadata...")
            metadata[file] = time.asctime(time.gmtime(time.time()))
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # save yaml file
    yaml_save_loc = os.path.join(project_folder, "train_prep.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
