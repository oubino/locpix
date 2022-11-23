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
remaining performance metrics."""

import yaml
import os
import numpy as np
from heptapods.preprocessing import datastruc
from heptapods.visualise.performance import plot_pr_curve, generate_conf_matrix
import heptapods.evaluate.metrics as metrics
from sklearn.metrics import precision_recall_curve, auc
import polars as pl
from datetime import datetime
import matplotlib.pyplot as plt
import pickle as pkl
from heptapods.visualise import vis_img

if __name__ == "__main__":

    # load yaml
    with open("recipes/img_seg/membrane_performance.yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # list items
    try:
        files = os.listdir(config["gt_files"])
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    fig_train, ax_train = plt.subplots()
    fig_test, ax_test = plt.subplots()

    linestyles = ["dashdot", "-", "--"]
    methods = ["classic", "cellpose", "ilastik"]

    for index, method in enumerate(methods):

        print(f"{method} ...")

        # get folder names
        seg_folder = config[method + "_seg_folder"]
        output_df_folder = config["output_" + method + "_df_folder"]
        output_seg_imgs = config["output_" + method + "_seg_imgs"]
        output_train_pr = config["output_train_" + method + "_pr"]
        output_test_pr = config["output_test_" + method + "_pr"]
        output_metrics = config["output_metrics_" + method]
        output_overlay_pr_curves = config["output_overlay_pr_curves"]
        output_conf_matrix = config["output_conf_matrix_" + method]

        # if output directory not present create it
        if not os.path.exists(output_df_folder):
            print("Making folder")
            os.makedirs(output_df_folder)

        # if output directory not present create it
        if not os.path.exists(output_seg_imgs):
            print("Making folder")
            os.makedirs(output_seg_imgs)

        # if output directory not present create it
        if not os.path.exists(output_train_pr):
            print("Making folder")
            os.makedirs(output_train_pr)

        # if output directory not present create it
        if not os.path.exists(output_test_pr):
            print("Making folder")
            os.makedirs(output_test_pr)

        # if output directory not present create it
        if not os.path.exists(output_metrics):
            print("Making folder")
            os.makedirs(output_metrics)

        # if output directory not present create it
        if not os.path.exists(output_overlay_pr_curves):
            print("Making folder")
            os.makedirs(output_overlay_pr_curves)

        # if output directory not present create it
        if not os.path.exists(output_conf_matrix):
            print("Making folder")
            os.makedirs(output_conf_matrix)

        print("Train set...")

        prob_list = np.array([])
        gt_list = np.array([])

        for file in files:

            if file.removesuffix(".parquet") not in config["train_files"]:
                continue

            print("File ", file)

            # load df
            item = datastruc.item(None, None, None, None)
            item.load_from_parquet(os.path.join(config["gt_files"], file))

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
        # print("pred", len(pred_list), pred_list)

        # calculate precision recall curve
        gt_list = gt_list.flatten()
        prob_list = prob_list.flatten()
        pr, rec, pr_threshold = precision_recall_curve(gt_list, prob_list, pos_label=1)
        baseline = len(gt[gt == 1]) / len(gt)

        # plot pr curve
        save_loc = os.path.join(output_train_pr, "_curve.pkl")
        plot_pr_curve(
            ax_train,
            method.capitalize(),
            linestyles[index],
            "darkorange",
            pr,
            rec,
            baseline,
            save_loc,
            pickle=True,
        )

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
            "train_set": config["train_files"],
            "test_set": config["test_files"],
            "threshold": threshold,
        }

        prob_list = np.array([])
        gt_list = np.array([])
        pred_list = np.array([])

        # sanity check all have same gt label map
        gt_label_map = None

        # threshold dataframe and save to parquet file with pred label
        for file in files:

            if file.removesuffix(".parquet") not in config["test_files"]:
                continue

            print("File ", file)

            item = datastruc.item(None, None, None, None)
            item.load_from_parquet(os.path.join(config["gt_files"], file))

            # load in histograms
            histo_loc = os.path.join(config["input_histo_folder"], item.name + ".pkl")
            with open(histo_loc, "rb") as f:
                histo = pkl.load(f)

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
            pred = save_df.select(pl.col("pred_label")).to_numpy()
            pred_list = np.append(pred_list, pred)


            # assign save dataframe to item
            item.df = save_df

            # save dataframe with predicted label column
            item.save_to_parquet(
                output_df_folder, drop_zero_label=False, drop_pixel_col=True
            )

            # also save image of predicted membrane
            output_img = np.where(img_prob > threshold, 1, 0)
            imgs = {key: value.T for key, value in histo.items()}

            # consider only zero channel
            save_loc = os.path.join(output_seg_imgs, item.name + ".png")
            vis_img.visualise_seg(
                imgs,
                output_img,
                item.bin_sizes,
                channels=[0],
                threshold=config["vis_threshold"],
                how=config["vis_interpolate"],
                origin="upper",
                blend_overlays=False,
                alpha_seg=0.8,
                cmap_seg=["k", "y"],
                save=True,
                save_loc=save_loc,
                four_colour=False,
            )

            # sanity check all have same gt label map
            if gt_label_map is None:
                gt_label_map = item.gt_label_map
            else:
                assert item.gt_label_map == gt_label_map

        # print("Sanity check... ")
        # print("gt", len(gt_list), gt_list)
        # print("pred", len(pred_list), pred_list)

        # calculate precision recall curve
        gt_list = gt_list.flatten()
        prob_list = prob_list.flatten()
        pr, rec, pr_threshold = precision_recall_curve(gt_list, prob_list, pos_label=1)
        baseline = len(gt[gt == 1]) / len(gt)

        # calculate confusion matrix
        date = datetime.today().strftime("%H_%M_%d_%m_%Y")
        saveloc = os.path.join(output_conf_matrix, f"conf_matrix_{date}.png")
        classes = [item.gt_label_map[0], item.gt_label_map[1]]
        pred_list = pred_list.flatten()
        generate_conf_matrix(gt_list, pred_list, classes, saveloc) 
        # could just use aggregated metric function to plot the confusion matrix

        # plot pr curve
        save_loc = os.path.join(output_test_pr, "_curve.pkl")
        plot_pr_curve(
            ax_test,
            method.capitalize(),
            linestyles[index],
            "darkorange",
            pr,
            rec,
            baseline,
            save_loc,
            pickle=True,
        )
        pr_auc = auc(rec, pr)
        add_metrics = {"pr_auc": pr_auc}

        # metric calculations based on final prediction
        save_loc = os.path.join(output_metrics, f"{date}.txt")
        metrics.aggregated_metric_calculation(
            output_df_folder,
            save_loc,
            gt_label_map,
            add_metrics=add_metrics,
            metadata=metadata,
        )

    fig_train.tight_layout()
    fig_test.tight_layout()

    # get handles and labels
    # handles_train, labels_train = ax_train.get_legend_handles_labels()
    # handles_test, labels_test = ax_test.get_legend_handles_labels()

    # specify order of items in legend
    # order = [1,0,2]

    # add legend to plot
    # ax_train.legend([handles_train[idx] for idx in order],[methods[idx] for idx in order])
    # ax_test.legend([handles_test[idx] for idx in order],[methods[idx] for idx in order])

    fig_train.savefig(os.path.join(output_overlay_pr_curves, "_train.png"), dpi=600)
    fig_test.savefig(os.path.join(output_overlay_pr_curves, "_test.png"), dpi=600)
