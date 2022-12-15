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
import tkinter as tk
from tkinter import filedialog


def main():

    parser = argparse.ArgumentParser(description="Ilastik prep")
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
            ilastik_prep_config.parse_config(config)
    else:
        config = ilastik_prep_config.config_gui()

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
