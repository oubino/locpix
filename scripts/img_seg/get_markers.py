#!/usr/bin/env python
"""Get markers from image
"""

import yaml
import os
from heptapods.preprocessing import datastruc
from heptapods.visualise import vis_img
from heptapods.img_processing import watershed
import numpy as np
import pickle as pkl


if __name__ == "__main__":

    # load yaml
    with open("recipes/img_seg/get_markers.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    try:
        files = os.listdir(config["input_folder"])
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    if not os.path.exists(config["markers_folder"]):
        print("Making folder")
        os.makedirs(config["markers_folder"])
    else:
        raise ValueError(
            "Will not get markers\
                         as there is already files in folder"
        )

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(config["input_folder"], file))

        # load in histograms
        histo_loc = os.path.join(config["input_histo_folder"], item.name + ".pkl")
        with open(histo_loc, "rb") as f:
            histo = pkl.load(f)

        # convert image to more visible form
        img = histo[0].T  # consider only the zero channel
        log_img = vis_img.manual_threshold(
            img, config["vis_threshold"], how=config["vis_interpolate"]
        )
        grey_log_img = vis_img.img_2_grey(log_img)  # convert img to grey

        markers = watershed.get_markers(grey_log_img)

        markers_loc = os.path.join(config["markers_folder"], item.name + ".npy")

        # save
        np.save(markers_loc, markers)
