#!/usr/bin/env python
"""Ilastik segmentation module

Process output of Ilastik
"""

import yaml
import os
from locpix.preprocessing import datastruc
from locpix.visualise import vis_img
import numpy as np
import pickle as pkl
import argparse
from locpix.scripts.img_seg import ilastik_output_config
import tkinter as tk
from tkinter import filedialog


def main():

    parser = argparse.ArgumentParser(description="Ilastik output")
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
            ilastik_output_config.parse_config(config)
    elif args.configgui:
        config = ilastik_output_config.config_gui()

    # list items
    input_folder = os.path.join(project_folder, "annotate/annotated")
    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    output_membrane_prob = os.path.join(
        project_folder, "ilastik/output/membrane/prob_map"
    )
    if os.path.exists(output_membrane_prob):
            raise ValueError(f"Cannot proceed as {output_membrane_prob} already exists")
    else:
        os.makedirs(output_membrane_prob)

    # if output directory not present create it
    output_cell_df = os.path.join(project_folder, "ilastik/output/cell/dataframe")
    if os.path.exists(output_cell_df):
            raise ValueError(f"Cannot proceed as {output_cell_df} already exists")
    else:
        os.makedirs(output_cell_df)

    # if output directory not present create it
    output_cell_img = os.path.join(project_folder, "ilastik/output/cell/img")
    if os.path.exists(output_cell_img):
            raise ValueError(f"Cannot proceed as {output_cell_img} already exists")
    else:
        os.makedirs(output_cell_img)

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(input_folder, file))

        # load in histograms
        input_histo_folder = os.path.join(project_folder, "annotate/histos")
        histo_loc = os.path.join(input_histo_folder, item.name + ".pkl")
        with open(histo_loc, "rb") as f:
            histo = pkl.load(f)

        # ---- membrane segmentation ----

        # load in ilastik_seg
        input_membrane_prob = os.path.join(
            project_folder, "ilastik/ilastik_output/membrane/prob_map_unprocessed"
        )
        membrane_prob_mask_loc = os.path.join(input_membrane_prob, item.name + ".npy")
        ilastik_seg = np.load(membrane_prob_mask_loc)

        # ilastik_seg is [y,x,c] where channel 0 is membranes, channel 1 is inside cells
        # given we only have labels for membranes and not inside cells
        # will currently ignore
        # chanel 1
        ilastik_seg = ilastik_seg[:, :, 0]

        # save the probability map
        prob_loc = os.path.join(output_membrane_prob, item.name + ".npy")
        np.save(prob_loc, ilastik_seg)

        # ---- cell segmentation ----

        # load in ilastik_seg
        input_cell_mask = os.path.join(
            project_folder, "ilastik/ilastik_output/cell/ilastik_output_mask"
        )
        cell_mask_loc = os.path.join(input_cell_mask, item.name + ".npy")
        ilastik_seg = np.load(cell_mask_loc)

        # ilastik_seg is [y,x,c] where channel 0 is segmentation
        # where each integer represents different instance of a cell
        # i.e. 1 = one cell; 2 = different cell; etc.
        ilastik_seg = ilastik_seg[:, :, 0]

        # save instance mask to dataframe
        df = item.mask_pixel_2_coord(ilastik_seg)
        item.df = df
        item.save_to_parquet(output_cell_df, drop_zero_label=False, drop_pixel_col=True)

        # save cell segmentation image - consider only zero channel
        imgs = {key: value.T for (key, value) in histo.items()}
        save_loc = os.path.join(output_cell_img, item.name + ".png")
        vis_img.visualise_seg(
            imgs,
            ilastik_seg,
            item.bin_sizes,
            channels=[0],
            threshold=config["vis_threshold"],
            how=config["vis_interpolate"],
            blend_overlays=True,
            alpha_seg=0.5,
            origin="upper",
            save=True,
            save_loc=save_loc,
            four_colour=True,
        )

        # save yaml file
        yaml_save_loc = os.path.join(project_folder, "ilastik_output.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
