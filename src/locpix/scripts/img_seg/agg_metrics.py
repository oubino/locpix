#!/usr/bin/env python
"""Module for calculating aggregated metric results"""

import os
import argparse
import json
import numpy as np

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

        # return method results in latex form
        latex_list = []
        latex_list.append(method)
        latex_list.append(f"{np.mean(recall_0):.2g}$\pm${np.std(recall_0):.2g}")
        latex_list.append(f"{np.mean(recall_1):.2g}$\pm${np.std(recall_1):.2g}")
        latex_list.append(f"{np.mean(iou_0):.2g}$\pm${np.std(iou_0):.2g}")
        latex_list.append(f"{np.mean(iou_1):.2g}$\pm${np.std(iou_1):.2g}")
        latex_list.append(f"{np.mean(macc_list):.2g}$\pm${np.std(macc_list):.2g}")
        latex_list.append(f"{np.mean(pr_auc_list):.2g}$\pm${np.std(pr_auc_list):.2g}")
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
    
