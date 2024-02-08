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
import argparse
import json
import time
from skimage.filters import threshold_otsu


def main():

    parser = argparse.ArgumentParser(description="Classic.")
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

    args = parser.parse_args()

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
        item = datastruc.item(None, None, None, None, None)
        item.load_from_parquet(os.path.join(input_folder, file))

        print("bin sizes", item.bin_sizes)

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
        log_img = vis_img.manual_threshold(
            img, config["vis_threshold"], how=config["vis_interpolate"]
        )
        grey_log_img = vis_img.img_2_grey(log_img)  # convert img to grey

        thresh = threshold_otsu(grey_log_img)
        semantic_mask = grey_log_img > thresh

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

        instance_mask = watershed.watershed_segment(
            grey_log_img, coords=markers
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

        # save cell segmentation image (as .npy) - consider only one channel
        output_cell_img = os.path.join(project_folder, "classic/cell/seg_img")
        save_loc = os.path.join(output_cell_img, item.name + ".npy")
        np.save(save_loc, instance_mask)

        # only plot the one channel specified
        # vis_img.visualise_seg(
        #     np.expand_dims(img, axis=0),
        #     instance_mask,
        #     item.bin_sizes,
        #     axes=[0],
        #     label_map=label_map,
        #     threshold=config["vis_threshold"],
        #     how=config["vis_interpolate"],
        #     origin="upper",
        #     blend_overlays=True,
        #     alpha_seg=0.5,
        #     save=True,
        #     save_loc=save_loc,
        #     four_colour=True,
        # )

        # save yaml file
        yaml_save_loc = os.path.join(project_folder, "classic.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
