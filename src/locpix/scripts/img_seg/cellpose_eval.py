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
from cellpose import models
import argparse
import json
import time


def main(*args):

    parser = argparse.ArgumentParser(description="Cellpose.")
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
        required=True,
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for preprocessing",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--project_metadata",
        action="store_true",
        help="check the metadata for the specified project and" "seek confirmation!",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        action="store",
        type=str,
        help="folder in project directory to save output",
        required=True,
    )
    parser.add_argument(
        "-u",
        "--user_model",
        action="store",
        type=str,
        help="The user model to load",
    )

    # so can be parsed either command line or as such
    if len(args) == 0:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args[0])

    print(args)

    project_folder = args.project_directory
    # load config
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

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
            print("Overwriting metadata...")
            metadata[file] = time.asctime(time.gmtime(time.time()))
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    # list items
    if config["test_files"] == "all":
        file_path = os.path.join(project_folder, "annotate/annotated")
        files = os.listdir(file_path)
        files = [os.path.join(file_path, file) for file in files]
    elif config["test_files"] == "metadata":
        with open(
            metadata_path,
        ) as file:
            metadata = json.load(file)
            test_files = metadata["test_files"]
            files = [
                os.path.join(project_folder, "annotate/annotated", file + ".parquet")
                for file in test_files
            ]
    else:
        files = [
            os.path.join(project_folder, "annotate/annotated", file + ".parquet")
            for file in config["test_files"]
        ]

    # output folder
    if args.output_folder is None:
        output_folder = "cellpose_no_train"
    else:
        output_folder = args.output_folder

    print("output folder", output_folder)

    # output directories
    output_membrane_prob = os.path.join(
        project_folder, f"{output_folder}/membrane/prob_map"
    )
    output_cell_df = os.path.join(
        project_folder, f"{output_folder}/cell/seg_dataframes"
    )
    output_cell_img = os.path.join(project_folder, f"{output_folder}/cell/seg_img")

    # if output directory not present create it
    output_directories = [output_membrane_prob, output_cell_df, output_cell_img]
    for directory in output_directories:
        if os.path.exists(directory):
            raise ValueError(f"Cannot proceed as {directory} already exists")
        else:
            os.makedirs(directory)

    for file in files:

        item = datastruc.item(None, None, None, None, None)
        item.load_from_parquet(file)

        # convert to histo
        histo, channel_map, label_map = item.render_histo(
            [config["channel"], config["alt_channel"]]
        )

        # ---- segment membranes ----
        if config["sum_chan"] is False:
            img = histo[0].T
        elif config["sum_chan"] is True:
            img = histo[0].T + histo[1].T
        else:
            raise ValueError("sum_chan should be true or false")
        img = vis_img.manual_threshold(
            img, config["img_threshold"], how=config["vis_interpolate"]
        )
        imgs = [img]

        if args.user_model is not None:
            model = models.CellposeModel(
                pretrained_model=args.user_model, gpu=config["use_gpu"]
            )

            # base model
            base_model = models.CellposeModel(model_type=config["model"])
            base_model = base_model.net.state_dict()
            sd = model.net.state_dict()
            for (k1, v1) in sd.items():
                vars = ["running_mean", "running_var", "num_batches_tracked"]
                # set these variables in our model to the ones from LC1
                if any(var in k1 for var in vars):
                    sd[k1] = base_model[k1]
            model.net.load_state_dict(sd)

            channels = config["channels"]
            # note diameter is set here may want to make user choice
            # doing one at a time (rather than in batch) like this might be very slow
            _, flows, _ = model.eval(
                imgs, diameter=config["diameter"], channels=channels
            )
        else:
            model = models.CellposeModel(
                model_type=config["model"], gpu=config["use_gpu"]
            )
            channels = config["channels"]
            # note diameter is set here may want to make user choice
            # doing one at a time (rather than in batch) like this might be very slow
            _, flows, _ = model.eval(
                imgs, diameter=config["diameter"], channels=channels
            )

        # get semantic mask
        # flows[0] as we have only one image so get first flow
        # flows[0][2] as this is the probability see
        # (https://cellpose.readthedocs.io/en/latest/api.html)
        semantic_mask = flows[0][2]
        lower = 1
        upper = 99
        X = semantic_mask
        x01 = np.percentile(X, lower)
        x99 = np.percentile(X, upper)
        X = (X - x01) / (x99 - x01)
        semantic_mask = np.clip(X, 0, 1)

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
            project_folder, f"{output_folder}/membrane/prob_map"
        )
        save_loc = os.path.join(output_membrane_prob, item.name + ".npy")
        np.save(save_loc, semantic_mask)

        # save markers
        np.save(markers_loc, markers)

        # save instance mask to dataframe
        df = item.mask_pixel_2_coord(instance_mask)
        item.df = df
        output_cell_df = os.path.join(
            project_folder, f"{output_folder}/cell/seg_dataframes"
        )
        item.save_to_parquet(output_cell_df, drop_zero_label=False, drop_pixel_col=True)

        # save cell segmentation image (as .npy) - consider only one channel
        output_cell_img = os.path.join(project_folder, f"{output_folder}/cell/seg_img")
        save_loc = os.path.join(output_cell_img, item.name + ".npy")
        np.save(save_loc, instance_mask)

        # save cell segmentation image
        # output_cell_img = os.path.join(project_folder,
        # f"{output_folder}/cell/seg_img")
        # save_loc = os.path.join(output_cell_img, item.name + ".png")

        # only plot the one channel specified
        # vis_img.visualise_seg(
        #     np.expand_dims(img, axis=0),
        #     instance_mask,
        #     item.bin_sizes,
        #     axes=[0],
        #     label_map=label_map,
        #     threshold=config["vis_threshold"],
        #     how=config["vis_interpolate"],
        #     blend_overlays=True,
        #     alpha_seg=0.5,
        #     origin="upper",
        #     save=True,
        #     save_loc=save_loc,
        #     four_colour=True,
        # )

        # save yaml file
        yaml_save_loc = os.path.join(project_folder, "cellpose_eval.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
