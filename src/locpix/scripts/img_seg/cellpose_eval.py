#!/usr/bin/env python
"""Cellpose segmentation module

Take in items and segment using Cellpose methods
"""

import yaml
import os
from locpix.preprocessing import datastruc
from locpix.visualise import vis_img
from locpix.img_processing import watershed
import numpy as np
import pickle as pkl
from cellpose import models
import argparse
from locpix.scripts.img_seg import cellpose_eval_config
import json
import time


def main():

    parser = argparse.ArgumentParser(
        description="Cellpose." "If no args are supplied will be run in GUI mode"
    )
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
    parser.add_argument(
        "-m",
        "--project_metadata",
        action="store_true",
        help="check the metadata for the specified project and" "seek confirmation!",
    )
    parser.add_argument(
        "-u",
        "--user_model",
        action="store_true",
        help="use the user model",
    )
    

    args = parser.parse_args()

    # if want to run in headless mode specify all arguments
    if args.project_directory is None and args.config is None:
        config, project_folder = cellpose_eval_config.config_gui()

    if args.project_directory is not None and args.config is None:
        parser.error(
            "If want to run in headless mode please supply arguments to"
            "config as well"
        )

    if args.config is not None and args.project_directory is None:
        parser.error(
            "If want to run in headless mode please supply arguments to project"
            "directory as well"
        )

    # headless mode
    if args.project_directory is not None and args.config is not None:
        project_folder = args.project_directory
        # load config
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            cellpose_eval_config.parse_config(config)

    metadata_path = os.path.join(project_folder, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # check metadata
        if args.project_metadata:
            print("".join([f"{key} : {value} \n" for key, value in metadata.items()]))
            check = input("Are you happy with this? (YES)")
            if check != "YES":
                exit()
        # add time ran this script to metadata
        file = os.path.basename(__file__)
        if file not in metadata:
            metadata[file] = time.asctime(time.gmtime(time.time()))
        else:
            print("Overwriting...")
            metadata[file] = time.asctime(time.gmtime(time.time()))
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # list items
    files = [os.path.join(project_folder, "annotate/annotated", file) for file in config["test_files"]]

    # output directories
    output_membrane_prob = os.path.join(project_folder, "cellpose/membrane/prob_map")
    output_cell_df = os.path.join(project_folder, "cellpose/cell/seg_dataframes")
    output_cell_img = os.path.join(project_folder, "cellpose/cell/seg_img")

    # if output directory not present create it
    output_directories = [output_membrane_prob, output_cell_df, output_cell_img]
    for directory in output_directories:
        if os.path.exists(directory):
            raise ValueError(f"Cannot proceed as {directory} already exists")
        else:
            os.makedirs(directory)

    for file in files:

        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(file + '.parquet')

        # conver to histo
        histo, axis_2_chan = item.render_histo()

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

        if args.user_model:
            model = models.CellposeModel(pretrained_model=config["user_model_path"])
            channels = config["channels"]
            # note diameter is set here may want to make user choice
            # doing one at a time (rather than in batch) like this might be very slow
            _, flows, _ = model.eval(imgs, channels=channels)
        else:
            model = models.CellposeModel(model_type=config["model"])
            channels = config["channels"]
            # note diameter is set here may want to make user choice
            # doing one at a time (rather than in batch) like this might be very slow
            _, flows, _ = model.eval(imgs, diameter=config["diameter"], channels=channels)
        # flows[0] as we have only one image so get first flow
        # flows[0][2] as this is the probability see
        # (https://cellpose.readthedocs.io/en/latest/api.html)
        semantic_mask = flows[0][2]

        # convert mask (probabilities) to range 0-1
        semantic_mask = (semantic_mask - np.min(semantic_mask)) / (
            np.max(semantic_mask) - np.min(semantic_mask)
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
            semantic_mask, coords=markers
        )  # watershed on the grey image

        # ---- save ----

        # save membrane mask
        output_membrane_prob = os.path.join(
            project_folder, "cellpose/membrane/prob_map"
        )
        save_loc = os.path.join(output_membrane_prob, item.name + ".npy")
        np.save(save_loc, semantic_mask)

        # save markers
        np.save(markers_loc, markers)

        # save instance mask to dataframe
        df = item.mask_pixel_2_coord(instance_mask)
        item.df = df
        output_cell_df = os.path.join(project_folder, "cellpose/cell/seg_dataframes")
        item.save_to_parquet(output_cell_df, drop_zero_label=False, drop_pixel_col=True)

        # save cell segmentation image - consider only zero channel
        imgs = {key: value.T for (key, value) in histo.items()}
        output_cell_img = os.path.join(project_folder, "cellpose/cell/seg_img")
        save_loc = os.path.join(output_cell_img, item.name + ".png")
        vis_img.visualise_seg(
            imgs,
            instance_mask,
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
        yaml_save_loc = os.path.join(project_folder, "cellpose_eval.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
