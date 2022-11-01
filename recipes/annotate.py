"""Annotate module

Take in items, convert to histograms, annotate,
visualise histo mask, save the exported annotation .parquet
"""

import yaml
import os
from heptapods.preprocessing import datastruc

if __name__ == "__main__":

    with open('recipes/annotate.yaml', "r") as ymlfile:
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

    if config['dim'] == 2:
        histo_size = (config['x_bins'], config['y_bins'])
    elif config['dim'] == 3:
        histo_size = (config['x_bins'], config['y_bins'], config['z_bins'])
    else:
        raise ValueError('Dim should be 2 or 3')

    for file in files:
        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(config['input_folder'], file))

        # coord2histo
        item.coord_2_histo(histo_size, plot=config['plot'],
                           vis_interpolation=config['vis_interpolation'])

        # manual segment
        item.manual_segment()

        # save df to parquet with mapping metadata
        item.save_to_parquet(os.path.join(config['output_folder'], 
                            item.name.replace('.csv', '.parquet')),
                            drop_zero_label=config['drop_zero_label'],
                            gt_label_map=config['gt_label_map'])
