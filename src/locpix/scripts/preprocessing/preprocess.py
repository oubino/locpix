#!/usr/bin/env python
"""Preprocessing module

Module takes in the .csv files and processes saving the datastructures
"""

import os
import yaml
from locpix.preprocessing import functions
import argparse
from locpix.scripts.preprocessing import preprocess_config
import time
import json
import socket


class project_info:
    """Project information metadata

    Attributes:
        metadata (dictionary) : Python dictionary containing
            the metadata"""

    def __init__(self, time, name):
        """Initialises metadata with args

        Args:
            time (string) : Time of project initialisation
            name (string) : Name of the project"""

        # dictionary
        self.metadata = {
            "machine": socket.gethostname(),
            "name": name,
            "init_time": time,
        }

    def save(self, path):
        """Save the dataframe as a .csv to
        the path

        Args:
            path (string) : Path to save to"""

        with open(path, "w") as outfile:
            json.dump(self.metadata, outfile)

    def load(self, path):
        """Load the dataframe from
        the path

        Args:
            path (string) : Path to load from"""

        self.metadata = json.load(path)


def main():

    # load path of .csv
    parser = argparse.ArgumentParser(
        description="Preprocess the data for\
        further processing."
        "If no args are supplied will be run in GUI mode"
    )

    parser.add_argument(
        "-i", "--input", action="store", type=str, help="path for the input data folder"
    )
    parser.add_argument(
        "-s",
        "--sanitycheck",
        action="store_true",
        help="whether to check correct csvs loaded in",
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
        "-o",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
    )

    args = parser.parse_args()

    # if want to run in headless mode specify all arguments
    if args.input is None and args.project_directory is None and args.config is None:
        config, csv_path, project_folder = preprocess_config.config_gui()
        print("csv path", csv_path)

    if args.input is not None and (
        args.project_directory is None or args.config is None
    ):
        parser.error(
            "If want to run in headless mode please supply arguments to project"
            "directory and config as well"
        )

    if args.project_directory is not None and (
        args.input is None or args.config is None
    ):
        parser.error(
            "If want to run in headless mode please supply arguments to input"
            "directory and config as well"
        )

    if args.config is not None and (args.input is None or args.config is None):
        parser.error(
            "If want to run in headless mode please supply arguments to project"
            "directory and input directory as well"
        )

    # headless mode
    if (
        args.input is not None
        and args.project_directory is not None
        and args.config is not None
    ):
        csv_path = args.input
        project_folder = args.project_directory
        # load config
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            preprocess_config.parse_config(config)

    # if output directory not present create it
    output_folder = os.path.join(project_folder, "preprocess/no_gt_label")
    if os.path.exists(os.path.join(project_folder, "preprocess")):
        raise ValueError(
            "You cannot choose this project folder"
            " as it already contains preprocessed data"
        )
    else:
        os.makedirs(output_folder)

        # initialise metadata and save
        metadata = project_info(time.asctime(time.gmtime(time.time())), project_folder)
        metadata.save(os.path.join(project_folder, "metadata.json"))

    # check with user
    print("List of csvs wich will be processed")
    csvs = [os.path.join(csv_path, f"{file}.csv") for file in config["include_files"]]
    print(csvs)
    if args.sanitycheck:
        check = input("If you are happy with these csvs type YES: ")
        if check != "YES":
            exit()

    # go through .csv -> convert to datastructure -> save
    for csv in csvs:
        item = functions.csv_to_datastruc(
            csv,
            config["dim"],
            config["channel_col"],
            config["frame_col"],
            config["x_col"],
            config["y_col"],
            config["z_col"],
            config["channel_choice"],
        )
        # have to not drop zero label
        # as no gt_label yet
        item.save_to_parquet(
            output_folder,
            drop_zero_label=False,
            drop_pixel_col=config["drop_pixel_col"],
        )

    # save yaml file
    config["input_data_folder"] = csv_path
    yaml_save_loc = os.path.join(project_folder, "preprocess.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
