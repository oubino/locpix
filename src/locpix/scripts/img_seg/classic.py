#!/usr/bin/env python
"""Classic segmentation module

Take in items and segment using classic methods
"""

import yaml
import os
from locpix.preprocessing import datastruc
from locpix.visualise import vis_img
from locpix.img_processing import watershed
import numpy as np
import pickle as pkl
import argparse
from locpix.scripts.img_seg import classic_config
import tkinter as tk
from tkinter import filedialog


def main():

    parser = argparse.ArgumentParser(description="Classic")
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

    # configuration folder
    if args.config is not None:
        # load yaml
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            classic_config.parse_config(config)
    else:
        config = classic_config.config_gui()

    # list items
    input_folder = os.path.join(project_folder, "annotate/annotated")
    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    output_membrane_prob = os.path.join(project_folder, "classic/membrane/prob_map")
    if os.path.exists(output_membrane_prob):
            raise ValueError(f"Cannot proceed as {output_membrane_prob} already exists")
    else:
        os.makedirs(output_membrane_prob)

    # if output directory not present create it
    output_cell_df = os.path.join(project_folder, "classic/cell/seg_dataframes")
    if os.path.exists(output_cell_df):
            raise ValueError(f"Cannot proceed as {output_cell_df} already exists")
    else:
        os.makedirs(output_cell_df)

    # if output directory not present create it
    output_cell_img = os.path.join(project_folder, "classic/cell/seg_img")
    if os.path.exists(output_cell_img):
            raise ValueError(f"Cannot proceed as {output_cell_img} already exists")
    else:
        os.makedirs(output_cell_img)

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(input_folder, file))

        print("bin sizes", item.bin_sizes)

        # load in histograms
        input_histo_folder = os.path.join(project_folder, "annotate/histos")
        histo_loc = os.path.join(input_histo_folder, item.name + ".pkl")
        with open(histo_loc, "rb") as f:
            histo = pkl.load(f)

        # ---- segment membranes ----
        if config["sum_chan"] is False:
            img = histo[0].T  # consider only the zero channel
        elif config["sum_chan"] is True:
            img = histo[0].T + histo[1].T
        else:
            raise ValueError("sum_chan should be true or false")
        log_img = vis_img.manual_threshold(
            img, config["vis_threshold"], how=config["vis_interpolate"]
        )
        grey_log_img = vis_img.img_2_grey(log_img)  # convert img to grey
        grey_img = vis_img.img_2_grey(img)  # convert img to grey

        # img mask
        semantic_mask = (grey_log_img - np.min(grey_log_img)) / (
            np.max(grey_log_img) - np.min(grey_log_img)
        )

        # ---- segment cells ----
        # get markers
        markers_loc = os.path.join(project_folder, "markers")
        markers_loc = os.path.join(markers_loc, item.name + ".npy")
        try:
            markers = np.load(markers_loc)
        except FileNotFoundError:
            raise ValueError(
                "Couldn't open the file/No markers were found in relevant location"
            )

        # tested very small amount annd line below is better than doing
        # watershed on grey_log_img
        instance_mask = watershed.watershed_segment(
            grey_img, coords=markers
        )  # watershed on the grey image

        # ---- save ----

        # save membrane mask
        output_membrane_prob = os.path.join(project_folder, "classic/membrane/prob_map")
        save_loc = os.path.join(output_membrane_prob, item.name + ".npy")
        np.save(save_loc, semantic_mask)

        # save markers
        np.save(markers_loc, markers)

        # save instance mask to dataframe
        df = item.mask_pixel_2_coord(instance_mask)
        item.df = df
        output_cell_df = os.path.join(project_folder, "classic/cell/seg_dataframes")
        item.save_to_parquet(output_cell_df, drop_zero_label=False, drop_pixel_col=True)

        imgs = {key: value.T for key, value in histo.items()}

        # save cell segmentation image - consider only zero channel
        output_cell_img = os.path.join(project_folder, "classic/cell/seg_img")
        save_loc = os.path.join(output_cell_img, item.name + ".png")
        vis_img.visualise_seg(
            imgs,
            instance_mask,
            item.bin_sizes,
            channels=[0],
            threshold=config["vis_threshold"],
            how=config["vis_interpolate"],
            origin="upper",
            blend_overlays=True,
            alpha_seg=0.5,
            save=True,
            save_loc=save_loc,
            four_colour=True,
        )

        # save yaml file
        yaml_save_loc = os.path.join(project_folder, "classic.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
