#!/usr/bin/env python
"""Module for calculating aggregated metric results"""

import os
import argparse
import json
import numpy as np
from locpix.preprocessing import datastruc
import polars as pl

def main():

    parser = argparse.ArgumentParser(
        description="Aggregated metrics"
    )
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        required=True,
        help="the location of the project directory",
    )

    args = parser.parse_args()
    project_folder = args.project_directory

    methods = ['classic', 'cellpose_no_train', 'cellpose_train', 'ilastik']

    metadata_path = os.path.join(project_folder, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # load in train and test files
        train_folds = metadata["train_folds"]
        val_folds = metadata["val_folds"]
        test_files = metadata["test_files"]

    # calculate aucprmin
    gt_file_path = os.path.join(project_folder, "annotate/annotated")
    files = os.listdir(gt_file_path)

    zero_tot = 0
    one_tot = 0
    
    for file in files:

        if file.removesuffix(".parquet") not in test_files:
            continue

        item = datastruc.item(None, None, None, None, None)
        item.load_from_parquet(os.path.join(gt_file_path, file))

        # count labels
        zeros = item.df["gt_label"].value_counts().filter(pl.col("gt_label") == 0).select(pl.col("counts"))[0,0]
        ones = item.df["gt_label"].value_counts().filter(pl.col("gt_label") == 1).select(pl.col("counts"))[0,0]

        zero_tot += zeros
        one_tot += ones

    skew = one_tot/(zero_tot + one_tot)
    aucprmin = 1 + ((1-skew)*np.log(1-skew))/skew

    # for every fold
    folds = len(metadata["train_folds"])

    for method in methods:

        iou_0 = []
        iou_1 = []
        recall_0 = []
        recall_1 = []
        macc_list = []
        miou_list = []
        pr_auc_list = []

        for fold in range(folds):

            #if method == 'classic' or method == 'cellpose_no_train':
            #    if fold != 0:
            #        continue
            
            folder = os.path.join(project_folder, f"membrane_performance/{method}/membrane/metrics/{fold}/")
            test_metrics_loc = os.listdir(folder)
            test_metrics_loc = [i for i in test_metrics_loc if i.startswith('test_')][0]
            test_metrics_loc = os.path.join(folder, test_metrics_loc) 

            # load in test file
            with open(test_metrics_loc, 'r') as f:
                for no, line in enumerate(f):
                    if no == 2:
                        iou_list = json.loads(line[11:])
                        iou_0.append(iou_list[0])
                        iou_1.append(iou_list[1])

                    elif no == 4:
                        recall_list = json.loads(line[14:])
                        recall_0.append(recall_list[0])
                        recall_1.append(recall_list[1])

                    elif no == 5:
                        macc = float(line[7:])
                        macc_list.append(macc)
                    
                    elif no == 6:
                        miou = float(line[7:])
                        miou_list.append(miou)

                    elif no == 8:
                        pr_auc = float(line[10:])
                        pr_auc_list.append(pr_auc)
        

        # return method results
        print('Method', method)
        print('Iou_0 ', np.mean(iou_0), ' +/- ', np.std(iou_0))
        print('Iou_1 ', np.mean(iou_1), ' +/- ', np.std(iou_1))
        print('recall_0 ', np.mean(recall_0), ' +/- ', np.std(recall_0))
        print('recall_1 ', np.mean(recall_1), ' +/- ', np.std(recall_1))
        print('macc ', np.mean(macc_list), ' +/- ', np.std(macc_list))
        print('miou ', np.mean(miou_list), ' +/- ', np.std(miou_list))
        print('pr_auc', np.mean(pr_auc_list), ' +/- ', np.std(pr_auc_list))
        auc = np.mean(pr_auc_list)
        pr_auc_norm = (auc - aucprmin)/(1 - aucprmin)
        pr_auc_norm_err = np.std(pr_auc_list)/(1-aucprmin)
        print('pr_auc_normalised', pr_auc_norm, ' +/- ', pr_auc_norm_err)

        # return method results in latex form
        latex_list = []
        latex_list.append(method)
        latex_list.append(f"{np.mean(recall_0):.2g}$\pm${np.std(recall_0):.2g}")
        latex_list.append(f"{np.mean(recall_1):.2g}$\pm${np.std(recall_1):.2g}")
        latex_list.append(f"{np.mean(iou_0):.2g}$\pm${np.std(iou_0):.2g}")
        latex_list.append(f"{np.mean(iou_1):.2g}$\pm${np.std(iou_1):.2g}")
        latex_list.append(f"{np.mean(macc_list):.2g}$\pm${np.std(macc_list):.2g}")
        latex_list.append(f"{np.mean(pr_auc_list):.2g}$\pm${np.std(pr_auc_list):.2g}")
        latex_list.append(f"{pr_auc_norm:.2g}$\pm${pr_auc_norm_err:.2g}")
        res = ' & '.join(latex_list)
        print(res)

    #iou_list (line3 : list of size 2)
    #recall lsit (line 5: list of size 2)
    #macc == accuraacy (line6 : int)
    #miou (line7 int)
    #pr_auc = line 9 int 
#
    #recall
    #iou
    #accuracy
    #pr auc


if __name__ == "__main__":
    main()
    
