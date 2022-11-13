"""Ilastik segmentation module

Process output of Ilastik
"""

import yaml
import os
from heptapods.preprocessing import datastruc
from heptapods.visualise import vis_img
import numpy as np
import pickle as pkl


if __name__ == "__main__":

    # load yaml
    with open("recipes/img_seg/ilastik.yaml", "r") as ymlfile:
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

        # ---- membrane segmentation ----

        # load in ilastik_seg
        membrane_prob_mask_loc = os.path.join(
            config["input_membrane_prob"], item.name + ".npy"
        )
        ilastik_seg = np.load(membrane_prob_mask_loc)

        # ilastik_seg is [y,x,c] where channel 0 is membranes, channel 1 is inside cells
        # given we only have labels for membranes and not inside cells will currently ignore
        # chanel 1
        ilastik_seg = ilastik_seg[:, :, 0]

        # save the probability map
        prob_loc = os.path.join(config["output_membrane_prob"], item.name + ".npy")
        np.save(prob_loc, ilastik_seg)

        # ---- cell segmentation ----

        # load in ilastik_seg
        cell_mask_loc = os.path.join(config["input_cell_mask"], item.name + ".npy")
        ilastik_seg = np.load(cell_mask_loc)

        # ilastik_seg is [y,x,c] where channel 0 is segmentation
        # where each integer represents different instance of a cell
        # i.e. 1 = one cell; 2 = different cell; etc.
        ilastik_seg = ilastik_seg[:, :, 0]

        # save instance mask to dataframe
        df = item.mask_pixel_2_coord(ilastik_seg)
        item.df = df
        item.save_to_parquet(
            config["output_cell_df"], drop_zero_label=False, drop_pixel_col=True
        )

        # save cell segmentation image - consider only zero channel
        imgs = {key: value.T for (key, value) in histo.items()}
        save_loc = os.path.join(config["output_cell_img"], item.name + ".png")
        vis_img.visualise_seg(
            imgs,
            ilastik_seg,
            item.bin_sizes,
            channels=[0],
            threshold=config["vis_threshold"],
            how=config["vis_interpolate"],
            origin="upper",
            save=True,
            save_loc=save_loc,
            four_colour=True,
        )
