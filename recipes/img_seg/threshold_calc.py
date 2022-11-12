"""For the specified files, it will calculate the performance
metrics over this training set

Calculate the threshold which maximises the performance

Then use this threshold for further testing"""

import yaml
import os
import numpy as np
from heptapods.preprocessing import datastruc
from sklearn.metrics import precision_recall_curve
import polars as pl

if __name__ == "__main__":

    # load yaml
    with open('recipes/img_seg/threshold_calc.yaml', "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    try:
        files = os.listdir(config['gt_files'])
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    pred_list = np.array([])
    gt_list = np.array([])

    for file in files:

        if not file in config['files']:
            pass

        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(config['gt_files'], file))
            
        # classic
        img_prob = np.load(os.path.join(config['classic_seg_folder'],
                                       item.name + '.npy'))
        merged_df = item.pred_pixel_2_coord(img_prob)
        merged_df = merged_df.rename({'pred_label': 'prob'})

        gt = merged_df.select(pl.col('gt_label')).to_numpy()
        pred = merged_df.select(pl.col('prob')).to_numpy()

        pred_list = np.append(pred_list, pred)
        gt_list = np.append(gt_list, gt)
    
    print('gt', len(gt_list), gt_list)
    print('pred', len(pred_list), pred_list)

    pr, rec, pr_threshold = precision_recall_curve(gt, 
                                                   pred, 
                                                   pos_label=1)

    max_index = np.argmax(rec)
    print('threshold', pr_threshold[max_index])

# then threshold files for classic ilastik and cellpose based on these to get output semantic seg

