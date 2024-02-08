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
import json
import time


def main():

    parser = argparse.ArgumentParser(description="Get markers.")
    # config_group = parser.add_mutually_exclusive_group(required=True)
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
                             for get markers",
        required=True,
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
        histo, channel_map, label_map = item.render_histo(
            [config["channel"], config["alt_channel"]]
        )

        # sum the histos
        img = histo[0].T + histo[1].T

        # convert image to more visible form
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
