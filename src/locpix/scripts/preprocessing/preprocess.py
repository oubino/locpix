#!/usr/bin/env python
"""Preprocessing module

Module takes in the .csv or .parquet files and processes saving the datastructures
"""

import os
import yaml
from locpix.preprocessing import functions
import argparse
import time
import json
import socket
import shutil


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

    # load path of .csv or .parquet
    parser = argparse.ArgumentParser(
        description="Preprocess the data for\
        further processing."
    )

    parser.add_argument(
        "-i",
        "--input",
        action="store",
        type=str,
        help="path for the input data folder",
        required=True,
    )
    parser.add_argument(
        "-s",
        "--sanitycheck",
        action="store_true",
        help="whether to check correct files loaded in",
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
        "-o",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--parquet_files",
        action="store_true",
        help="if true will process as parquet files",
    )

    args = parser.parse_args()

    input_path = args.input
    project_folder = args.project_directory
    # load config
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # if output directory not present create it
    output_folder = os.path.join(project_folder, "preprocess/no_gt_label")
    if not os.path.exists(os.path.join(project_folder, "preprocess")):
        os.makedirs(output_folder)

        # initialise metadata and save
        metadata = project_info(time.asctime(time.gmtime(time.time())), project_folder)
        metadata.save(os.path.join(project_folder, "metadata.json"))

    # if all is specified then consider all files otherwise consider specified files
    if config["include_files"] == "all":
        include_files = os.listdir(args.input)
        include_files = [os.path.splitext(item)[0] for item in include_files]
    else:
        include_files = config["include_files"]

    # check with user
    print("List of files which will be processed")
    if args.parquet_files is False:
        files = [os.path.join(input_path, f"{file}.csv") for file in include_files]
        # check file not already present
        for file in files:
            file_name = os.path.basename(file)
            output_path = os.path.join(
                output_folder, f"{file_name.replace('.csv', '.parquet')}"
            )
            input("check this makes sense and if it doesnt then stop")
            if os.path.exists(output_path):
                raise ValueError("Can't preprocess as output file already exists")
        print(files)
        if args.sanitycheck:
            check = input("If you are happy with these csvs type YES: ")
            if check != "YES":
                exit()
    elif args.parquet_files is True:
        files = [os.path.join(input_path, f"{file}.parquet") for file in include_files]
        # check file not already present
        for file in files:
            file_name = os.path.basename(file)
            output_path = os.path.join(output_folder, f"{file_name}")
            if os.path.exists(output_path):
                raise ValueError("Can't preprocess as output file already exists")
        print(files)
        if args.sanitycheck:
            check = input("If you are happy with these parquets type YES: ")
            if check != "YES":
                exit()

    # go through files -> convert to datastructure -> save
    for file in files:
        if args.parquet_files is False:
            file_type = "csv"
        elif args.parquet_files is True:
            file_type = "parquet"
        item = functions.file_to_datastruc(
            file,
            file_type,
            config["dim"],
            config["channel_col"],
            config["frame_col"],
            config["x_col"],
            config["y_col"],
            config["z_col"],
            channel_choice=config["channel_choice"],
            channel_label=config["channel_label"],
        )
        # have to not drop zero label
        # as no gt_label yet
        item.save_to_parquet(
            output_folder,
        )

    # add visualisation notebook
    shutil.copyfile(
        "src/locpix/templates/visualisation.ipynb",
        os.path.join(project_folder, "visualisation.ipynb"),
    )

    # save yaml file
    config["input_data_folder"] = input_path
    yaml_save_loc = os.path.join(
        project_folder, f"preprocess_{os.path.basename(input_path)}.yaml"
    )
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
