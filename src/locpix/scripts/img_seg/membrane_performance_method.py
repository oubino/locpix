#!/usr/bin/env python
"""Module for evluating performance of membrane
segmentation

For the train set, it will use the probability map over
the image, and use this to assign probability of belonging
to membrane to each localisation.
Then it will use this to plot precision-recall curve,
and calculate the optimal threshold.

For the test set, it will use the probability maps to
plot a precision recall curve.
It will then use the threshold calculated from the train
set to threshold the test images

Now we have output membrane segmentations on the test set
which we can use to finally calculate the other
remaining performance metrics.

This just takes in one method"""

import yaml
import os
import numpy as np
from locpix.preprocessing import datastruc

# from locpix.visualise.performance import plot_pr_curve , generate_binary_conf_matrix
import locpix.evaluate.metrics as metrics
from sklearn.metrics import precision_recall_curve, auc
import polars as pl
from datetime import datetime

# import matplotlib.pyplot as plt
# from locpix.visualise import vis_img
import argparse
import json
import time


def main():

    parser = argparse.ArgumentParser(
        description="Membrane performance metrics on data."
    )
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
        required=True,
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
        "--method",
        action="store",
        type=str,
        help="the method used",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--project_metadata",
        action="store_true",
        help="check the metadata for the specified project and seek confirmation!",
    )

    args = parser.parse_args()
    project_folder = args.project_directory
    # load config
    with open(args.config, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # define method
    method = args.method

    metadata_path = os.path.join(project_folder, "metadata.json")
    with open(
        metadata_path,
    ) as file:
        metadata = json.load(file)
        # check metadata
        if args.project_metadata:
            print("".join([f"{key} : {value} \n" for key, value in metadata.items()]))
            check = input("Are you happy with this? (YES)")
            if check != "YES":
                exit()
        # add time ran this script to metadata
        file = os.path.basename(__file__)
        if file not in metadata:
            metadata[file] = time.asctime(time.gmtime(time.time()))
        else:
            print("Overwriting metadata...")
            metadata[file] = time.asctime(time.gmtime(time.time()))
        # load in train and test files
        train_folds = metadata["train_folds"]
        val_folds = metadata["val_folds"]
        test_files = metadata["test_files"]
        with open(metadata_path, "w") as outfile:
            json.dump(metadata, outfile)

    for fold, (train_files, val_files) in enumerate(zip(train_folds, val_folds)):

        # check files
        if not set(train_files).isdisjoint(test_files):
            raise ValueError("Train files and test files shared files!!")
        if not set(train_files).isdisjoint(val_files):
            raise ValueError("Train files and test files shared files!!")
        if not set(val_files).isdisjoint(test_files):
            raise ValueError("Train files and test files shared files!!")
        if len(set(train_files)) != len(train_files):
            raise ValueError("Train files contains duplicates")
        if len(set(val_files)) != len(val_files):
            raise ValueError("Test files contains duplicates")
        if len(set(test_files)) != len(test_files):
            raise ValueError("Test files contains duplicates")

        # list items
        gt_file_path = os.path.join(project_folder, "annotate/annotated")
        try:
            files = os.listdir(gt_file_path)
        except FileNotFoundError:
            raise ValueError("There should be some files to open")

        # one date for all methods
        date = datetime.today().strftime("%H_%M_%d_%m_%Y")

        print(f"{method} ...")

        # get folder names

        seg_folder = os.path.join(project_folder, f"{method}/{fold}/membrane/prob_map/")

        output_df_folder_test = os.path.join(
            project_folder,
            f"membrane_performance/{method}/membrane/seg_dataframes/test/{fold}",
        )
        output_df_folder_val = os.path.join(
            project_folder,
            f"membrane_performance/{method}/membrane/seg_dataframes/val/{fold}",
        )
        # output_seg_imgs_test = os.path.join(
        #    project_folder,
        #    f"membrane_performance/{method}/membrane/seg_images/test/{fold}",
        # )
        # output_seg_imgs_val = os.path.join(
        #    project_folder,
        #    f"membrane_performance/{method}/membrane/seg_images/val/{fold}",
        # )
        # output_train_pr = os.path.join(
        #    project_folder,
        #    f"membrane_performance/{method}/membrane/train_pr/{fold}",
        # )
        # output_test_pr = os.path.join(
        #    project_folder, f"membrane_performance/{method}/membrane/test_pr/
        # {fold}"
        # )
        # output_val_pr = os.path.join(
        #    project_folder, f"membrane_performance/{method}/membrane/val_pr/{fold}"
        # )
        output_metrics = os.path.join(
            project_folder, f"membrane_performance/{method}/membrane/metrics/{fold}"
        )
        # output_conf_matrix = os.path.join(
        #    project_folder,
        #    f"membrane_performance/{method}/membrane/conf_matrix/{fold}",
        # )

        # create output folders
        folders = [
            output_df_folder_test,
            output_df_folder_val,
            # output_seg_imgs_val,
            # output_seg_imgs_test,
            # output_train_pr,
            # output_val_pr,
            # output_test_pr,
            output_metrics,
            # output_conf_matrix,
        ]

        for folder in folders:
            # if output directory not present create it
            if os.path.exists(folder):
                raise ValueError(f"Cannot proceed as {folder} already exists")
            else:
                os.makedirs(folder)

        print("Train set...")

        prob_list = np.array([])
        gt_list = np.array([])

        for file in files:

            if file.removesuffix(".parquet") not in train_files:
                continue

            print("File ", file)

            # load df
            item = datastruc.item(None, None, None, None, None)
            item.load_from_parquet(os.path.join(gt_file_path, file))

            # load prob map
            img_prob = np.load(os.path.join(seg_folder, item.name + ".npy"))

            # merge prob map and df

            merged_df = item.mask_pixel_2_coord(img_prob)
            merged_df = merged_df.rename({"pred_label": "prob"})

            # get gt and predicted probability
            gt = merged_df.select(pl.col("gt_label")).to_numpy()
            prob = merged_df.select(pl.col("prob")).to_numpy()

            # append to aggregated data set
            prob_list = np.append(prob_list, prob)
            gt_list = np.append(gt_list, gt)

        # print("Sanity check... ")
        # print("gt", len(gt_list), gt_list)

        # calculate precision recall curve
        gt_list = gt_list.flatten()
        prob_list = prob_list.flatten().round(2)
        pr, rec, pr_threshold = precision_recall_curve(gt_list, prob_list, pos_label=1)
        baseline = len(gt_list[gt_list == 1]) / len(gt_list)

        # pr, recall saved for train
        save_loc = os.path.join(output_metrics, f"train_{date}.txt")
        lines = ["Overall results", "-----------"]
        lines.append(f"prcurve_pr: {list(pr)}")
        lines.append(f"prcurve_rec: {list(rec)}")
        lines.append(f"prcurve_baseline: {baseline}")
        with open(save_loc, "w") as f:
            f.writelines("\n".join(lines))

        # plot pr curve
        # save_loc = os.path.join(output_train_pr, "_curve.pkl")
        # plot_pr_curve(
        #     ax_train,
        #     method.capitalize(),
        #     linestyles[index],
        #     "darkorange",
        #     pr,
        #     rec,
        #     baseline,
        #     # save_loc,
        #     # pickle=True,
        # )

        # calculate optimal threshold
        if config["maximise_choice"] == "recall":
            max_index = np.argmax(rec)
        elif config["maximise_choice"] == "f":
            # get f measure and find optimal
            fmeas = (2 * pr * rec) / (pr + rec)
            # remove nan from optimal threshold calculation
            nan_array = np.isnan(fmeas)
            fmeas = np.where(nan_array, 0, fmeas)
            max_index = np.argmax(fmeas)
        else:
            raise ValueError("Choice not supported")

        threshold = pr_threshold[max_index]
        print("Threshold for ", method, " :", threshold)

        print("Test set...")

        metadata = {
            "train_set": train_files,
            "test_set": test_files,
            "threshold": threshold,
            "val_set": val_files,
        }

        prob_list = np.array([])
        gt_list = np.array([])

        # sanity check all have same gt label map
        gt_label_map = None

        # threshold dataframe and save to parquet file with pred label
        for file in files:

            if file.removesuffix(".parquet") not in test_files:
                continue

            print("File ", file)

            item = datastruc.item(None, None, None, None, None)
            item.load_from_parquet(os.path.join(gt_file_path, file))

            # convert to histo
            histo, channel_map, label_map = item.render_histo(
                [config["channel"], config["alt_channel"]]
            )

            # load prob map
            img_prob = np.load(os.path.join(seg_folder, item.name + ".npy"))

            # merge prob map and df
            merged_df = item.mask_pixel_2_coord(img_prob)
            merged_df = merged_df.rename({"pred_label": "prob"})

            # get gt and predicted probability
            gt = merged_df.select(pl.col("gt_label")).to_numpy()
            prob = merged_df.select(pl.col("prob")).to_numpy()

            save_df = merged_df.select(
                [
                    pl.all(),
                    pl.when(pl.col("prob") > threshold)
                    .then(1)
                    .otherwise(0)
                    .alias("pred_label"),
                ]
            )

            # append to aggregated data set
            prob_list = np.append(prob_list, prob)
            gt_list = np.append(gt_list, gt)

            # assign save dataframe to item
            item.df = save_df

            # save dataframe with predicted label column
            item.save_to_parquet(
                output_df_folder_test, drop_zero_label=False, drop_pixel_col=True
            )

            # also save image of predicted membrane
            # output_img = np.where(img_prob > threshold, 1, 0)

            # img = np.transpose(histo, (0, 2, 1))

            # consider the correct channel
            # save_loc = os.path.join(output_seg_imgs_test, item.name + ".png")
            # vis_img.visualise_seg(
            #     img,
            #     output_img,
            #     item.bin_sizes,
            #     axes=[0],
            #     label_map=label_map,
            #     threshold=config["vis_threshold"],
            #     how=config["vis_interpolate"],
            #     origin="upper",
            #     blend_overlays=False,
            #     alpha_seg=0.8,
            #     cmap_seg=["k", "y"],
            #     save=True,
            #     save_loc=save_loc,
            #     four_colour=False,
            # )

            # sanity check all have same gt label map
            if gt_label_map is None:
                gt_label_map = item.gt_label_map
            else:
                assert item.gt_label_map == gt_label_map

        # print("Sanity check... ")
        # print("gt", len(gt_list), gt_list)

        # calculate precision recall curve
        gt_list = gt_list.flatten()
        prob_list = prob_list.flatten().round(2)
        pr, rec, pr_threshold = precision_recall_curve(gt_list, prob_list, pos_label=1)
        baseline = len(gt_list[gt_list == 1]) / len(gt_list)

        # plot pr curve
        # save_loc = os.path.join(output_test_pr, "_curve.pkl")
        # plot_pr_curve(
        #     ax_test,
        #     method.capitalize(),
        #     linestyles[index],
        #     "darkorange",
        #     pr,
        #     rec,
        #     baseline,
        #     # save_loc,
        #     # pickle=True,
        # )
        pr_auc = auc(rec, pr)
        add_metrics = {
            "pr_auc": pr_auc,
            "prcurve_pr": list(pr),
            "prcurve_rec": list(rec),
            "prcurve_baseline": baseline,
        }

        # metric calculations based on final prediction
        save_loc = os.path.join(output_metrics, f"test_{date}.txt")
        _ = metrics.aggregated_metrics(
            output_df_folder_test,
            save_loc,
            gt_label_map,
            add_metrics=add_metrics,
            metadata=metadata,
        )

        # calculate confusion matrix
        # saveloc = os.path.join(output_conf_matrix, f"conf_matrix_test_{date}.png")
        # classes = [item.gt_label_map[0], item.gt_label_map[1]]
        # generate_binary_conf_matrix(tn, fp, fn, tp, classes, saveloc)
        # could just use aggregated metric function to plot the confusion matrix

        print("Val set...")

        prob_list = np.array([])
        gt_list = np.array([])

        # sanity check all have same gt label map
        gt_label_map = None

        # threshold dataframe and save to parquet file with pred label
        for file in files:

            if file.removesuffix(".parquet") not in val_files:
                continue

            print("File ", file)

            item = datastruc.item(None, None, None, None, None)
            item.load_from_parquet(os.path.join(gt_file_path, file))

            # convert to histo
            histo, channel_map, label_map = item.render_histo(
                [config["channel"], config["alt_channel"]]
            )

            # load prob map
            img_prob = np.load(os.path.join(seg_folder, item.name + ".npy"))

            # merge prob map and df
            merged_df = item.mask_pixel_2_coord(img_prob)
            merged_df = merged_df.rename({"pred_label": "prob"})

            # get gt and predicted probability
            gt = merged_df.select(pl.col("gt_label")).to_numpy()
            prob = merged_df.select(pl.col("prob")).to_numpy()

            save_df = merged_df.select(
                [
                    pl.all(),
                    pl.when(pl.col("prob") > threshold)
                    .then(1)
                    .otherwise(0)
                    .alias("pred_label"),
                ]
            )

            # append to aggregated data set
            prob_list = np.append(prob_list, prob)
            gt_list = np.append(gt_list, gt)

            # assign save dataframe to item
            item.df = save_df

            # save dataframe with predicted label column
            item.save_to_parquet(
                output_df_folder_val, drop_zero_label=False, drop_pixel_col=True
            )

            # also save image of predicted membrane
            # output_img = np.where(img_prob > threshold, 1, 0)

            # img = np.transpose(histo, (0, 2, 1))

            # consider the correct channel
            # save_loc = os.path.join(output_seg_imgs_val, item.name + ".png")
            # vis_img.visualise_seg(
            #     img,
            #     output_img,
            #     item.bin_sizes,
            #     axes=[0],
            #     label_map=label_map,
            #     threshold=config["vis_threshold"],
            #     how=config["vis_interpolate"],
            #     origin="upper",
            #     blend_overlays=False,
            #     alpha_seg=0.8,
            #     cmap_seg=["k", "y"],
            #     save=True,
            #     save_loc=save_loc,
            #     four_colour=False,
            # )

            # sanity check all have same gt label map
            if gt_label_map is None:
                gt_label_map = item.gt_label_map
            else:
                assert item.gt_label_map == gt_label_map

            # print("Sanity check... ")
            # print("gt", len(gt_list), gt_list)

        # calculate precision recall curve
        gt_list = gt_list.flatten()
        prob_list = prob_list.flatten().round(2)
        pr, rec, pr_threshold = precision_recall_curve(gt_list, prob_list, pos_label=1)
        baseline = len(gt_list[gt_list == 1]) / len(gt_list)

        # plot pr curve
        # save_loc = os.path.join(output_val_pr, "_curve.pkl")
        # plot_pr_curve(
        #     ax_val,
        #     method.capitalize(),
        #     linestyles[index],
        #     "darkorange",
        #     pr,
        #     rec,
        #     baseline,
        #     # save_loc,
        #     # pickle=True,
        # )
        pr_auc = auc(rec, pr)
        add_metrics = {
            "pr_auc": pr_auc,
            "prcurve_pr": list(pr),
            "prcurve_rec": list(rec),
            "prcurve_baseline": baseline,
        }

        # metric calculations based on final prediction
        save_loc = os.path.join(output_metrics, f"val_{date}.txt")
        _ = metrics.aggregated_metrics(
            output_df_folder_val,
            save_loc,
            gt_label_map,
            add_metrics=add_metrics,
            metadata=metadata,
        )

        # calculate confusion matrix
        # saveloc = os.path.join(output_conf_matrix, f"conf_matrix_val_{date}.png")
        # classes = [item.gt_label_map[0], item.gt_label_map[1]]
        # generate_binary_conf_matrix(tn, fp, fn, tp, classes, saveloc)

        # fig_train.tight_layout()
        # fig_test.tight_layout()
        # fig_val.tight_layout()

        # get handles and labels
        # handles_train, labels_train = ax_train.get_legend_handles_labels()
        # handles_test, labels_test = ax_test.get_legend_handles_labels()

        # specify order of items in legend
        # order = [1,0,2]

        # add legend to plot
        # ax_train.legend([handles_train[idx] for idx in order],
        #                   [methods[idx] for idx in order])
        # ax_test.legend([handles_test[idx] for idx in order],
        #                   [methods[idx] for idx in order])

        # fig_train.savefig(os.path.join(output_overlay_pr_curves,
        # "_train.png"), dpi=600)
        # fig_test.savefig(os.path.join(output_overlay_pr_curves,
        # "_test.png"), dpi=600)
        # fig_val.savefig(os.path.join(output_overlay_pr_curves,
        # "_val.png"), dpi=600)

    # save yaml file
    yaml_save_loc = os.path.join(project_folder, "membrane_performance.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
