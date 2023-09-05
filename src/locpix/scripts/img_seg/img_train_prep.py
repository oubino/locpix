#!/usr/bin/env python
"""Prepare images for training module

Take in items and convert to images ready for training
"""

import os
import json

from locpix.preprocessing import datastruc
import numpy as np

# import skimage
from locpix.visualise import vis_img
import tifffile


def preprocess_train_files(project_folder, config, metadata, fold, model_name):
    """Preprocess data

    Args:
        project_folder (string) : Project folder
        config (dict) : Configuration for the
            train script
        metadata (dict) : Metadata associated with
            the project
        fold (int) : Fold to be preprocessed"""

    # check train val test files
    train_files = metadata["train_folds"][fold]
    val_files = metadata["val_folds"][fold]
    test_files = metadata["test_files"]
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
    print("Train files")
    print(train_files)
    print("Test files")
    print(test_files)
    print("Val files")
    print(val_files)

    # list items
    input_root = os.path.join(project_folder, "annotate/annotated")
    try:
        files = os.listdir(input_root)
        files = [os.path.splitext(file)[0] for file in files]
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # make necessary folders if not present
    train_folder = os.path.join(project_folder, f"train_files/{model_name}/train")
    val_folder = os.path.join(project_folder, f"train_files/{model_name}/val")
    folders = [
        train_folder,
        val_folder,
    ]
    for folder in folders:
        if os.path.exists(folder):
            raise ValueError(f"Cannot proceed as {folder} already exists")
        else:
            os.makedirs(folder)

    # convert files into imgs and masks
    train_files = [os.path.join(input_root, file + ".parquet") for file in train_files]
    val_files = [os.path.join(input_root, file + ".parquet") for file in val_files]
    parquet_2_img(
        train_files,
        config["labels"],
        config["sum_chan"],
        config["img_threshold"],
        config["img_interpolate"],
        train_folder,
    )
    parquet_2_img(
        val_files,
        config["labels"],
        config["sum_chan"],
        config["img_threshold"],
        config["img_interpolate"],
        val_folder,
    )


def preprocess_all_files(project_folder, config, metadata, model_name):
    """Preprocess data

    Args:
        project_folder (string) : Project folder
        config (dict) : Configuration for the
            train script
        metadata (dict) : Metadata associated with
            the project
        fold (int) : Fold to be preprocessed"""

    # check train val test files
    train_files = metadata["train_files"]
    test_files = metadata["test_files"]
    # check files
    if not set(train_files).isdisjoint(test_files):
        raise ValueError("Train files and test files shared files!!")
    if len(set(train_files)) != len(train_files):
        raise ValueError("Train files contains duplicates")
    if len(set(test_files)) != len(test_files):
        raise ValueError("Test files contains duplicates")
    all_files = train_files + test_files
    print("All files")
    print(all_files)

    # list items
    input_root = os.path.join(project_folder, "annotate/annotated")
    try:
        files = os.listdir(input_root)
        files = [os.path.splitext(file)[0] for file in files]
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # make necessary folders if not present
    folder = os.path.join(project_folder, f"train_files/{model_name}/all")
    if os.path.exists(folder):
        raise ValueError(f"Cannot proceed as {folder} already exists")
    else:
        os.makedirs(folder)

    # convert files into imgs and masks
    all_files = [os.path.join(input_root, file + ".parquet") for file in all_files]
    parquet_2_img(
        all_files,
        config["labels"],
        config["sum_chan"],
        config["img_threshold"],
        config["img_interpolate"],
        folder,
    )


# def preprocess_test_files(project_folder, config, metadata):
#     """Preprocess data
#
#     Args:
#         project_folder (string) : Project folder
#         config (dict) : Configuration for the
#             train script
#         metadata (dict) : Metadata associated with
#             the project"""
#
#     folds = len(metadata["train_folds"])
#     for fold in range(folds):
#         # check train val test files
#         train_files = metadata["train_folds"][fold]
#         val_files = metadata["val_folds"][fold]
#         test_files = metadata["test_files"]
#         # check files
#         if not set(train_files).isdisjoint(test_files):
#             raise ValueError("Train files and test files shared files!!")
#         if not set(train_files).isdisjoint(val_files):
#             raise ValueError("Train files and val files shared files!!")
#         if not set(val_files).isdisjoint(test_files):
#             raise ValueError("Val files and test files shared files!!")
#         if len(set(train_files)) != len(train_files):
#             raise ValueError("Train files contains duplicates")
#         if len(set(val_files)) != len(val_files):
#             raise ValueError("Val files contains duplicates")
#         if len(set(test_files)) != len(test_files):
#             raise ValueError("Test files contains duplicates")
#         print("Fold: ", fold)
#         print("Train files")
#         print(train_files)
#         print("Test files")
#         print(test_files)
#         print("Val files")
#         print(val_files)
#
#     # list items
#     input_root = os.path.join(project_folder, "annotate/annotated")
#     try:
#         files = os.listdir(input_root)
#         files = [os.path.splitext(file)[0] for file in files]
#     except FileNotFoundError:
#         raise ValueError("There should be some files to open")
#
#     # make necessary folders if not present
#     test_folder = os.path.join(project_folder, f"test_files/{model_name}/")
#     folders = [
#         test_folder,
#     ]
#     for folder in folders:
#         if os.path.exists(folder):
#             raise ValueError(f"Cannot proceed as {folder} already exists")
#         else:
#             os.makedirs(folder)
#
#     # convert files into imgs and masks
#     test_files = [os.path.join(input_root, file + ".parquet") for file in test_files]
#     parquet_2_img(
#         test_files,
#         config["labels"],
#         config["sum_chan"],
#         config["img_threshold"],
#         config["img_interpolate"],
#         test_folder,
#     )


def clean_up(project_folder, model_name):
    """Clean up data

    Args:
        project_folder (string) : Project folder"""

    train_folder = os.path.join(project_folder, f"train_files/{model_name}/train")
    val_folder = os.path.join(project_folder, f"train_files/{model_name}/val")

    # remove train files
    for file in os.listdir(train_folder):
        file_path = os.path.join(train_folder, file)
        os.remove(file_path)

    os.rmdir(train_folder)

    # remove val files
    for file in os.listdir(val_folder):
        file_path = os.path.join(val_folder, file)
        os.remove(file_path)

    os.rmdir(val_folder)


def clean_up_all(project_folder, model_name):
    """Clean up data

    Args:
        project_folder (string) : Project folder"""

    img_folder = os.path.join(project_folder, f"train_files/{model_name}/all")

    # remove train files
    for file in os.listdir(img_folder):
        file_path = os.path.join(img_folder, file)
        os.remove(file_path)

    os.rmdir(img_folder)


def parquet_2_img(files, labels, sum_chan, img_threshold, img_interpolate, folder):
    """Convert data from .parquet files to .png files
    and save

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

        img_info_path = os.path.join(folder, "img_info.json")
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
        img = vis_img.manual_threshold(img, img_threshold, how=img_interpolate)

        # save
        # img_folder = os.path.join(folder, 'imgs')
        # label_folder = os.path.join(folder, 'labels')
        # folders = [img_folder, label_folder]
        # for folder in folders:
        img_path = os.path.join(folder, item.name + ".tif")
        label_path = os.path.join(folder, item.name + "_masks.tif")

        # save
        old_img = img
        old_label = label
        tifffile.imwrite(img_path, img)
        tifffile.imwrite(label_path, label)
        # plt.imsave(img_path, img)
        # plt.imsave(label_path, label)
        # load and check
        img = tifffile.imread(img_path)
        label = tifffile.imread(label_path)
        np.testing.assert_array_equal(old_img, img)
        np.testing.assert_array_equal(old_label, label)
