#!/usr/bin/env python
"""Preprocessing module

Module takes in the .csv files and processes saving the datastructures
"""

import os
import yaml
from locpix.preprocessing import functions
import argparse
import tkinter as tk
from tkinter import filedialog
from locpix.scripts.preprocessing import preprocess_config


def main():

    # load path of .csv
    parser = argparse.ArgumentParser(
        description="Preprocess the data for\
        further processing"
    )
    # data_group.add_argument(
    #    "-e",
    #    "--env",
    #    action="store",
    #    type=str,
    #    help="location of .env file for data path",
    # )
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

    # if args.env is not None:
    #    dotenv_path = ".env"
    #    dotenv.load_dotenv(dotenv_path)
    #    csv_path = os.getenv("RAW_DATA_PATH")

    # input data folder
    if args.input is not None:
        csv_path = args.input
    else:
        root = tk.Tk()
        root.withdraw()
        csv_path = filedialog.askdirectory(title="Data folder")

    # project folder
    if args.project_directory is not None:
        project_folder = args.project_directory
    else:
        root = tk.Tk()
        root.withdraw()
        project_folder = filedialog.askdirectory(title="Project folder")

    # if output directory not present create it
    output_folder = os.path.join(project_folder, "preprocess/no_gt_label")
    if os.path.exists(output_folder):
        raise ValueError(
            "You cannot choose this project folder"
            " as it already contains preprocessed data"
        )
    else:
        os.makedirs(output_folder)

    # list all .csv at input
    csvs = os.listdir(csv_path)
    csvs = [csv for csv in csvs if csv.endswith(".csv")]

    # load in configuration .yaml
    if args.config is not None:
        # load yaml
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            preprocess_config.parse_config(config)
    else:
        files = [csv.removesuffix(".csv") for csv in csvs]
        config = preprocess_config.config_gui(files)

    # remove excluded files
    csvs = [csv for csv in csvs if csv.removesuffix(".csv") in config["include_files"]]

    # check with user
    print("List of csvs wich will be processed")
    print(csvs)
    if args.sanitycheck:
        check = input("If you are happy with these csvs type YES: ")
        if check != "YES":
            exit()

    # go through .csv -> convert to datastructure -> save
    for csv in csvs:
        input_file = os.path.join(csv_path, csv)
        item = functions.csv_to_datastruc(
            input_file,
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
