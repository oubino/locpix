#!/usr/bin/env python
"""Annotate module

Take in items, convert to histograms, annotate,
visualise histo mask, save the exported annotation .parquet
"""

import yaml
import os
from heptapods.preprocessing import datastruc
from heptapods.visualise import vis_img
import pickle as pkl

if __name__ == "__main__":

    with open("recipes/annotate.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    try:
        files = os.listdir(config["input_folder"])
    except FileNotFoundError:
        raise ValueError("There should be some preprocessed files to open")

    # if output directory not present create it
    if not os.path.exists(config["output_folder"]):
        print("Making folder")
        os.makedirs(config["output_folder"])

    # if output directory for seg imgs not present create it
    if not os.path.exists(config["output_seg_folder"]):
        print("Making folder")
        os.makedirs(config["output_seg_folder"])

    # if output directory for seg imgs not present create it
    if not os.path.exists(config["histo_folder"]):
        print("Making folder")
        os.makedirs(config["histo_folder"])

    if config["dim"] == 2:
        histo_size = (config["x_bins"], config["y_bins"])
    elif config["dim"] == 3:
        histo_size = (config["x_bins"], config["y_bins"], config["z_bins"])
    else:
        raise ValueError("Dim should be 2 or 3")

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(config["input_folder"], file))

        # coord2histo
        item.coord_2_histo(histo_size, vis_interpolation=config["vis_interpolation"])

        # manual segment
        item.manual_segment()

        # save df to parquet with mapping metadata
        item.save_to_parquet(
            config["output_folder"],
            drop_zero_label=config["drop_zero_label"],
            gt_label_map=config["gt_label_map"],
        )

        # save histogram
        save_loc = os.path.join(config["histo_folder"], item.name + ".pkl")
        with open(save_loc, "wb") as f:
            pkl.dump(item.histo, f)

        # save images
        if config["save_img"] is True:
            save_loc = config["output_seg_folder"]
            img_dict = item.get_img_dict()
            save_loc = os.path.join(config["output_seg_folder"], item.name + ".png")
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
