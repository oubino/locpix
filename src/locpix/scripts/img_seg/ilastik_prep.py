#!/usr/bin/env python
"""Ilastik prep module

Take in items and prepare for Ilastik
"""

import yaml
import os
from locpix.visualise import vis_img
import numpy as np
import pickle as pkl
import argparse
from locpix.scripts.img_seg import ilastik_prep_config
import json
import time

def main():

    parser = argparse.ArgumentParser(description="Ilastik prep."\
        "If no args are supplied will be run in GUI mode"
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
        type=str,
        help="check the metadata for the specified project and"\
             "seek confirmation!"
    )

    args = parser.parse_args()

    # if want to run in headless mode specify all arguments
    if args.project_directory is None and args.config is None:
        config, project_folder = ilastik_prep_config.config_gui()

    if args.project_directory is not None and args.config is None:
        parser.error("If want to run in headless mode please supply arguments to"\
                     "config as well")

    if args.config is not None and args.project_directory is None:
        parser.error("If want to run in headless mode please supply arguments to project"\
                     "directory as well")

    # headless mode
    if args.project_directory is not None and args.config is not None:
        project_folder = args.project_directory
        # load config
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            ilastik_prep_config.parse_config(config)

    metadata_path = os.path.join(project_folder,'metadata.json')
    with open(metadata_path,) as file:
        metadata = json.load(file)
        # check metadata
        if args.project_metadata:
            print("".join([f"{key} : {value} \n" for key, value in metadata.items()]))
            check = input('Are you happy with this? (YES)')
            if check != "YES":
                exit()
        # add time ran this script to metadata
        file = os.path.basename(__file__)
        if file not in metadata:
            metadata[file] = time.asctime(time.gmtime(time.time()))
        else:
            print('Overwriting...')
            metadata[file] = time.asctime(time.gmtime(time.time()))
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # list items
    input_histo_folder = os.path.join(project_folder, "annotate/histos")
    try:
        files = os.listdir(input_histo_folder)
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    output_folder = os.path.join(project_folder, "ilastik/prep")
    if os.path.exists(output_folder):
        raise ValueError(f"Cannot proceed as {output_folder} already exists")
    else:
        os.makedirs(output_folder)

    for file in files:

        histo_loc = os.path.join(input_histo_folder, file)

        # load in histograms
        with open(histo_loc, "rb") as f:
            histo = pkl.load(f)

        img_list = []
        for histo in histo.values():
            img = histo.T
            log_img = vis_img.manual_threshold(
                img, config["threshold"], how=config["interpolation"]
            )
            img_list.append(vis_img.img_2_grey(log_img))
        img = np.stack(img_list, axis=2)

        # all images are saved in yxc
        file_name = file.removesuffix(".pkl")
        save_loc = os.path.join(output_folder, file_name + ".npy")
        np.save(save_loc, img)

    # save yaml file
    yaml_save_loc = os.path.join(project_folder, "ilastik_prep.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
