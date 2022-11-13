"""Ilastik prep module

Take in items and prepare for Ilastik
"""

import yaml
import os
from heptapods.visualise import vis_img
import numpy as np
import pickle as pkl


if __name__ == "__main__":

    # load yaml
    with open("recipes/img_seg/ilastik.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)["preprocessing"]

    # list items
    try:
        files = os.listdir(config["input_histo_folder"])
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # if output directory not present create it
    if not os.path.exists(config["output_folder"]):
        print("Making folder")
        os.makedirs(config["output_folder"])

    for file in files:

        histo_loc = os.path.join(config["input_histo_folder"], file)

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
        save_loc = os.path.join(config["output_folder"], file_name + ".npy")
        np.save(save_loc, img)
