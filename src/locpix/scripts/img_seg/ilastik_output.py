#!/usr/bin/env python
"""Ilastik segmentation module

Process output of Ilastik
"""

import yaml
import os
from locpix.preprocessing import datastruc

# from locpix.visualise import vis_img
import numpy as np
import argparse
import json
import time


def main():

    parser = argparse.ArgumentParser(description="Ilastik output.")
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

    # need to iterate over folds

    for fold in range(5):

        # if output directory not present create it
        output_membrane_prob = os.path.join(
            project_folder, f"ilastik/output/membrane/prob_map/{fold}"
        )
        if os.path.exists(output_membrane_prob):
            raise ValueError(f"Cannot proceed as {output_membrane_prob} already exists")
        else:
            os.makedirs(output_membrane_prob)

        # if output directory not present create it
        output_cell_df = os.path.join(
            project_folder, f"ilastik/output/cell/dataframe/{fold}"
        )
        if os.path.exists(output_cell_df):
            raise ValueError(f"Cannot proceed as {output_cell_df} already exists")
        else:
            os.makedirs(output_cell_df)

        # if output directory not present create it
        output_cell_img = os.path.join(
            project_folder, f"ilastik/output/cell/img/{fold}"
        )
        if os.path.exists(output_cell_img):
            raise ValueError(f"Cannot proceed as {output_cell_img} already exists")
        else:
            os.makedirs(output_cell_img)

        for file in files:
            item = datastruc.item(None, None, None, None, None)
            item.load_from_parquet(os.path.join(input_folder, file))

            # convert to histo
            # histo, channel_map, label_map = item.render_histo(
            #     [config["channel"], config["alt_channel"]]
            # )

            # ---- membrane segmentation ----

            # load in ilastik_seg
            input_membrane_prob = os.path.join(
                project_folder, f"ilastik/ilastik_pixel/{fold}"
            )
            membrane_prob_mask_loc = os.path.join(
                input_membrane_prob, item.name + ".npy"
            )
            ilastik_seg = np.load(membrane_prob_mask_loc)

            # ilastik_seg is [y,x,c] where channel 0 is membranes,
            # channel 1 is inside cells
            # given we only have labels for membranes and not inside cells
            # will currently ignore
            # chanel 1
            ilastik_seg = ilastik_seg[:, :, 0]

            # save the probability map
            prob_loc = os.path.join(output_membrane_prob, item.name + ".npy")
            np.save(prob_loc, ilastik_seg)

            # ---- cell segmentation ----

            # load in ilastik_seg
            input_cell_mask = os.path.join(
                project_folder, f"ilastik/ilastik_boundary/{fold}"
            )
            cell_mask_loc = os.path.join(input_cell_mask, item.name + ".npy")
            ilastik_seg = np.load(cell_mask_loc)

            # ilastik_seg is [y,x,c] where channel 0 is segmentation
            # where each integer represents different instance of a cell
            # i.e. 1 = one cell; 2 = different cell; etc.
            ilastik_seg = ilastik_seg[:, :, 0]

            # save instance mask to dataframe
            df = item.mask_pixel_2_coord(ilastik_seg)
            item.df = df
            item.save_to_parquet(
                output_cell_df, drop_zero_label=False, drop_pixel_col=True
            )

            # save cell segmentation image (as .npy) - consider only one channel
            output_cell_img = os.path.join(
                project_folder, f"ilastik/output/cell/img/{fold}"
            )
            save_loc = os.path.join(output_cell_img, item.name + ".npy")
            np.save(save_loc, ilastik_seg)

            # save cell segmentation image - consider only one channel
            # img = np.transpose(histo, (0, 2, 1))
            # save_loc = os.path.join(output_cell_img, item.name + ".png")
            # vis_img.visualise_seg(
            #     img,
            #     ilastik_seg,
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
        yaml_save_loc = os.path.join(project_folder, "ilastik_output.yaml")
        with open(yaml_save_loc, "w") as outfile:
            yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
