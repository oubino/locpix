"""Cellpose segmentation module

Take in items and segment using Cellpose methods
"""

import yaml
import os
from heptapods.preprocessing import datastruc
from heptapods.visualise import vis_img
from heptapods.img_processing import watershed
import numpy as np
import pickle as pkl
from cellpose import models


if __name__ == "__main__":

    # load yaml
    with open("recipes/img_seg/cellpose.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    try:
        files = os.listdir(config["input_folder"])
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    if not os.path.exists(config["output_membrane_prob"]):
        print("Making folder")
        os.makedirs(config["output_membrane_prob"])

    # if output directory not present create it
    if not os.path.exists(config["output_cell_df"]):
        print("Making folder")
        os.makedirs(config["output_cell_df"])

    # if output directory not present create it
    if not os.path.exists(config["output_cell_img"]):
        print("Making folder")
        os.makedirs(config["output_cell_img"])

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(config["input_folder"], file))

        # load in histograms
        histo_loc = os.path.join(config["input_histo_folder"], item.name + ".pkl")
        with open(histo_loc, "rb") as f:
            histo = pkl.load(f)

        # ---- segment membranes ----

        if config["sum_chan"] is False:
            img = histo[0].T  # consider only the zero channel
        elif config["sum_chan"] is True:
            img = histo[0].T + histo[1].T
        else:
            raise ValueError("sum_chan should be true or false")
        img = vis_img.manual_threshold(
            img, config["vis_threshold"], how=config["vis_interpolate"]
        )
        imgs = [img]
        model = models.CellposeModel(model_type=config["model"])
        channels = config["channels"]
        # note diameter is set here may want to make user choice
        # doing one at a time (rather than in batch) like this might be very slow
        _, flows, _ = model.eval(imgs, diameter=config["diameter"], channels=channels)
        semantic_mask = flows[0][2]

        # convert mask (probabilities) to range 0-1
        semantic_mask = (semantic_mask - np.min(semantic_mask)) / (
            np.max(semantic_mask) - np.min(semantic_mask)
        )

        # ---- segment cells ----
        # get markers
        markers_loc = os.path.join(config["markers_loc"], item.name + ".npy")
        try:
            markers = np.load(markers_loc)
        except FileNotFoundError:
            raise ValueError(
                "Couldn't open the file/No markers were found in relevant location"
            )

        # tested very small amount annd line below is better than doing watershed on grey_log_img
        instance_mask = watershed.watershed_segment(
            semantic_mask, coords=markers
        )  # watershed on the grey image

        # ---- save ----

        # save membrane mask
        save_loc = os.path.join(config["output_membrane_prob"], item.name + ".npy")
        np.save(save_loc, semantic_mask)

        # save markers
        np.save(markers_loc, markers)

        # save instance mask to dataframe
        df = item.mask_pixel_2_coord(instance_mask)
        item.df = df
        item.save_to_parquet(
            config["output_cell_df"], drop_zero_label=False, drop_pixel_col=True
        )

        # save cell segmentation image - consider only zero channel
        imgs = {key: value.T for (key, value) in histo.items()}
        save_loc = os.path.join(config["output_cell_img"], item.name + ".png")
        vis_img.visualise_seg(
            imgs,
            instance_mask,
            item.bin_sizes,
            channels=[0],
            threshold=config["vis_threshold"],
            how=config["vis_interpolate"],
            origin="upper",
            save=True,
            save_loc=save_loc,
            four_colour=True,
        )
