#!/usr/bin/env python
"""Get markers from image
"""

import yaml
import os
from locpix.preprocessing import datastruc
from locpix.visualise import vis_img
from locpix.img_processing import watershed
import numpy as np
import pickle as pkl
import argparse
from locpix.scripts.img_seg import get_markers_config
import tkinter as tk
from tkinter import filedialog


def main():

    parser = argparse.ArgumentParser(description="Get markers")
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

    args = parser.parse_args()

    # input project directory
    if args.project_directory is not None:
        project_folder = args.project_directory
    else:
        root = tk.Tk()
        root.withdraw()
        project_folder = filedialog.askdirectory(title="Project directory")

    if args.config is not None:
        # load yaml
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            get_markers_config.parse_config(config)
    else:
        config = get_markers_config.config_gui()

    # list items
    input_folder = os.path.join(project_folder, "preprocess/no_gt_label")
    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    markers_folder = os.path.join(project_folder, "markers")
    if not os.path.exists(markers_folder):
        print("Making folder")
        os.makedirs(markers_folder)
    else:
        raise ValueError(
            "Will not get markers\
                         as there is already files in folder"
        )

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(input_folder, file))

        # load in histograms
        input_histo_folder = os.path.join(project_folder, "annotate/histos")
        histo_loc = os.path.join(input_histo_folder, item.name + ".pkl")
        with open(histo_loc, "rb") as f:
            histo = pkl.load(f)

        # convert image to more visible form
        img = histo[0].T  # consider only the zero channel
        log_img = vis_img.manual_threshold(
            img, config["vis_threshold"], how=config["vis_interpolate"]
        )
        grey_log_img = vis_img.img_2_grey(log_img)  # convert img to grey

        markers = watershed.get_markers(grey_log_img)

        markers_loc = os.path.join(markers_folder, item.name + ".npy")

        # save
        np.save(markers_loc, markers)

    # save yaml file
    yaml_save_loc = os.path.join(project_folder, "get_markers.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
