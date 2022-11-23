#!/usr/bin/env python
"""Preprocessing module

Module takes in the .csv files and processes saving the datastructures
"""

import dotenv
import os
import yaml
from heptapods.preprocessing import functions
import argparse
import tkinter as tk
from tkinter import filedialog
from . import preprocess_config

if __name__ == "__main__":

    # load path of .csv 
    parser = argparse.ArgumentParser(description='Preprocess the data for\
        further processing')
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument('-e', '--env', action='store', type=str,
                        help='location of .env file for data path')
    data_group.add_argument('-g', '--gui', action='store_true',
                        help='whether to use gui to get data path')
    data_group.add_argument('-f', '--folder', action='store', type=str,
                        help='path for the data folder')
    parser.add_argument('-s', '--sanitycheck', action='store_true',
                        help='whether to check correct csvs loaded in')
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument('-c', '--config', action='store', type=str,
                        help='the location of the .yaml configuaration file\
                             for preprocessing')
    config_group.add_argument('-cg', '--configgui', action='store_true',
                        help='whether to use gui to get the configuration')
    
    args = parser.parse_args()
    
    if args.env is not None:
        dotenv_path = ".env"
        dotenv.load_dotenv(dotenv_path)
        csv_path = os.getenv("RAW_DATA_PATH")
    
    elif args.gui:
        root = tk.Tk()
        root.withdraw()
        csv_path = filedialog.askdirectory()

    elif args.folder is not None:
        csv_path = args.folder

    # list all .csv in this location
    csvs = os.listdir(csv_path)
    csvs = [csv for csv in csvs if csv.endswith(".csv")]

    if args.config is not None:
        # load yaml
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            preprocess_config.parse_config(config)
    elif args.configgui:
        preprocess_config.config_gui('output/preprocess/preprocess.yaml')

    # remove excluded files
    csvs = [
        csv for csv in csvs if csv.removesuffix(".csv") not in config["exclude_files"]
    ]

    # check with user
    print('List of csvs wich will be processed')
    print(csvs)
    if args.sanitycheck:
        check = input('If you are happy with these csvs type YES: ')
        if check != 'YES':
            exit()

    # if output directory not present create it
    if not os.path.exists(config["output_folder"]):
        os.makedirs(config["output_folder"])

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
            config["output_folder"],
            drop_zero_label=False,
            drop_pixel_col=config["drop_pixel_col"],
        )
