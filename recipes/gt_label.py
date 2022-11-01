"""Gt label generating recipe

This generates the gt label in the dataframe, 
and checks/makes assumption that the gt label
isn't already there
"""

import os
import yaml
from heptapods.preprocessing import datastruc
import polars as pl

def gt_label_generator(df):
    """Custom function takes in a polars dataframe and adds a 
    gt_label column in whichever user specified way
        
        Args:
            df (polars dataframe) : Dataframe with localisations"""
    
    # this just takes the channel column as the ground truth label
    df = df.with_column((pl.col('channel')).alias('gt_label'))

    return df

if __name__ == "__main__":

    # load yaml
    with open('recipes/gt_label.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    try:
        files = os.listdir(config['input_folder'])
    except FileNotFoundError:
        raise ValueError("There should be some preprocessed files to open")

    # if output directory not present create it
    if not os.path.exists(config['output_folder']):
        print('Making folder')
        os.makedirs(config['output_folder'])

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(config['input_folder'], file))

        # check no gt label already present
        if 'gt_label' in item.df.columns:
            raise ValueError('Manual segment cannot be called on a file which\
                              already has gt labels in it')

        # generate gt label
        item.df = gt_label_generator(item.df)

        # save df to parquet with mapping metadata
        # note drop zero label important is False as we have 
        # channel 0 (EGFR) -> gt_label 0 -> don't want to drop this
        # drop pixel col is False as we by this point have 
        # no pixel col
        item.save_to_parquet(os.path.join(config['output_folder'],
                             item.name.replace('.csv', '.parquet')),
                             drop_zero_label=False,
                             drop_pixel_col=False,
                             gt_label_map=config['gt_label_map'])

