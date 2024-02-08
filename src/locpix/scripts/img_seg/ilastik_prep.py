#!/usr/bin/env python
"""Ilastik prep module

Take in items and prepare for Ilastik
"""

import yaml
import os
from locpix.visualise import vis_img
from locpix.preprocessing import datastruc
import numpy as np
import argparse
import json
import time


def main():

    parser = argparse.ArgumentParser(description="Ilastik prep.")
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
                             for preprocessing",
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

    # list items
    input_folder = os.path.join(project_folder, "annotate/annotated")
    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    output_folder = os.path.join(project_folder, "ilastik/prep")

    img_folder = os.path.join(output_folder, "imgs")
    if os.path.exists(img_folder):
        raise ValueError(f"Cannot proceed as {img_folder} already exists")
    else:
        os.makedirs(img_folder)

    mask_folder = os.path.join(output_folder, "masks")
    if os.path.exists(mask_folder):
        raise ValueError(f"Cannot proceed as {mask_folder} already exists")
    else:
        os.makedirs(mask_folder)

    for file in files:

        item = datastruc.item(None, None, None, None, None)
        item.load_from_parquet(os.path.join(input_folder, file))

        # ---------- images -----------

        # conver to histo
        histo, channel_map, label_map = item.render_histo(config["channels"])

        # ilastik needs channel last and need to tranpose histogram
        # for image space
        img = np.transpose(histo, (2, 1, 0))

        img = vis_img.manual_threshold(
            img, config["threshold"], how=config["interpolation"]
        )
        img = vis_img.img_2_grey(img)

        # all images are saved in yxc
        file_name = file.removesuffix(".parquet")
        save_loc = os.path.join(output_folder, "imgs", file_name + ".npy")
        np.save(save_loc, img)

        # --------- masks -------------

        # label histo
        label_histo = item.render_seg()

        # label img
        # need to tranpose histogram for image space
        label_img = np.transpose(label_histo, (1, 0))

        # in ilastik
        # 0: no label
        # 1: membrane
        # 2: not membrane
        # we mean
        # 0: not membrane
        # 1: membrane
        # therefore need to set all 0s to 2s
        label_img = np.where(label_img == 0, 2, label_img)
        label_img = np.expand_dims(label_img, axis=2)
        label_img = label_img.astype("uint8")

        # separate into bkg and label
        bkg = np.where(label_img == 2, label_img, 0)
        label = np.where(label_img == 1, label_img, 0)

        # undersample the background as otherwise hard to
        # load into Ilastik
        rng = np.random.default_rng()
        noise = rng.choice(2, size=label_img.shape, p=[0.8, 0.2])
        bkg = np.where(noise == 0, 0, bkg)

        # combine bkg back with label
        label_img = label + bkg

        # save masks
        file_name = file.removesuffix(".parquet")
        save_loc = os.path.join(output_folder, "masks", file_name + ".npy")
        np.save(save_loc, label_img)

    # save yaml file
    yaml_save_loc = os.path.join(project_folder, "ilastik_prep.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
