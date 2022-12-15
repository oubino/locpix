#!/usr/bin/env python
"""Annotate module

Take in items, convert to histograms, annotate,
visualise histo mask, save the exported annotation .parquet
"""

import yaml
import os
from locpix.preprocessing import datastruc
from locpix.visualise import vis_img
import pickle as pkl
import argparse
from locpix.scripts.preprocessing import annotate_config
import tkinter as tk
from tkinter import filedialog


def main():

    parser = argparse.ArgumentParser(description="Annotate the data")
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
            annotate_config.parse_config(config)
    else:
        config = annotate_config.config_gui()

    # list items
    input_folder = os.path.join(project_folder, "preprocess/no_gt_label")
    print(input_folder)
    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        raise ValueError("There should be some preprocessed files to open")

    # if output directory not present create it
    output_folder = os.path.join(project_folder, "annotate/annotated")
    if os.path.exists(output_folder):
        raise ValueError(f"Cannot proceed as {output_folder} already exists")
    else:
        os.makedirs(output_folder)

    # if output directory for seg imgs not present create it
    output_seg_folder = os.path.join(project_folder, "annotate/seg_imgs")
    if os.path.exists(output_seg_folder):
        raise ValueError(f"Cannot proceed as {output_seg_folder} already exists")
    else:
        os.makedirs(output_seg_folder)

    # if output directory for seg imgs not present create it
    histo_folder = os.path.join(project_folder, "annotate/histos")
    if os.path.exists(histo_folder):
        raise ValueError(f"Cannot proceed as {histo_folder} already exists")
    else:
        os.makedirs(histo_folder)

    if config["dim"] == 2:
        histo_size = (config["x_bins"], config["y_bins"])
    elif config["dim"] == 3:
        histo_size = (config["x_bins"], config["y_bins"], config["z_bins"])
    else:
        raise ValueError("Dim should be 2 or 3")

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(input_folder, file))

        # coord2histo
        item.coord_2_histo(histo_size, vis_interpolation=config["vis_interpolation"])

        # manual segment
        item.manual_segment()

        # save df to parquet with mapping metadata
        item.save_to_parquet(
            output_folder,
            drop_zero_label=config["drop_zero_label"],
            gt_label_map=config["gt_label_map"],
        )

        # save histogram
        save_loc = os.path.join(histo_folder, item.name + ".pkl")
        with open(save_loc, "wb") as f:
            pkl.dump(item.histo, f)

        # save images
        if config["save_img"] is True:
            save_loc = output_seg_folder
            img_dict = item.get_img_dict()
            save_loc = os.path.join(output_seg_folder, item.name + ".png")
            vis_img.visualise_seg(
                img_dict,
                item.histo_mask.T,
                item.bin_sizes,
                channels=config["vis_channels"],
                threshold=config["save_threshold"],
                how=config["save_interpolate"],
                alphas=config["alphas"],
                blend_overlays=False,
                alpha_seg=config["alpha_seg"],
                cmap_img=None,
                cmap_seg=config["cmap_seg"],
                figsize=config["fig_size"],
                origin="upper",
                save=True,
                save_loc=save_loc,
                four_colour=config["four_colour"],
                background_one_colour=config["background_one_colour"],
            )

    # save yaml file
    yaml_save_loc = os.path.join(project_folder, "annotate.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
