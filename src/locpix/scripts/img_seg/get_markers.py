#!/usr/bin/env python
"""Get markers from image
"""

import yaml
import os
from locpix.preprocessing import datastruc
from locpix.visualise import vis_img
from locpix.img_processing import watershed
import numpy as np
import argparse
from locpix.scripts.img_seg import get_markers_config
import json
import time


def main():

    parser = argparse.ArgumentParser(
        description="Get markers." "If no args are supplied will be run in GUI mode"
    )
    # config_group = parser.add_mutually_exclusive_group(required=True)
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
                             for get markers",
    )
    parser.add_argument(
        "-m",
        "--project_metadata",
        action="store_true",
        help="check the metadata for the specified project and" "seek confirmation!",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="if true then will overwrite files"
    )

    args = parser.parse_args()

    # if want to run in headless mode specify all arguments
    if args.project_directory is None and args.config is None:
        config, project_folder = get_markers_config.config_gui()

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
            get_markers_config.parse_config(config)

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

    # list items
    input_folder = os.path.join(project_folder, "annotate/annotated")
    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    markers_folder = os.path.join(project_folder, "markers")
    if not os.path.exists(markers_folder):
        os.makedirs(markers_folder)

    for file in files:
        item = datastruc.item(None, None, None, None, None)
        item.load_from_parquet(os.path.join(input_folder, file))

        # save location
        markers_loc = os.path.join(markers_folder, item.name + ".npy")
        if os.path.exists(markers_loc) and not args.force:
            continue

        # convert to histo
        histo, channel_map, label_map = item.render_histo([config["channel"]])

        # convert image to more visible form
        img = histo[0].T
        log_img = vis_img.manual_threshold(
            img, config["vis_threshold"], how=config["vis_interpolate"]
        )
        grey_log_img = vis_img.img_2_grey(log_img)  # convert img to grey

        markers = watershed.get_markers(grey_log_img)

        # save
        np.save(markers_loc, markers)

    # save yaml file
    yaml_save_loc = os.path.join(project_folder, "get_markers.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
