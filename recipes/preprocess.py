"""Preprocessing module

Module takes in the .csv files and processes saving the datastructures
"""

import dotenv
import os
import yaml
from heptapods.preprocessing import functions

if __name__ == "__main__":

    # load path of .csv
    dotenv_path = '.env'
    dotenv.load_dotenv(dotenv_path)
    csv_path = os.getenv("RAW_DATA_PATH")

    # load yaml
    with open('recipes/preprocess.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list all .csv in this location
    csvs = os.listdir(csv_path)
    csvs = [csv for csv in csvs if csv.endswith('.csv')]

    # if output directory not present create it
    if not os.path.exists(config['output_folder']):
        os.makedirs(config['output_folder'])

    # go through .csv -> convert to datastructure -> save
    for csv in csvs:
        input_file = os.path.join(csv_path, csv)
        item = functions.csv_to_datastruc(input_file, config['dim'],
                                          config['channel_col'],
                                          config['frame_col'],
                                          config['x_col'],
                                          config['y_col'],
                                          config['z_col'],
                                          config['channel_choice'])

        # have to not drop zero label
        # as no gt_label yet
        item.save_to_parquet(config['output_folder'],
                             drop_zero_label=False,
                             drop_pixel_col=config['drop_pixel_col'])
