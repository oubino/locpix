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
from locpix.preprocessing import datastruc
from locpix.visualise.performance import plot_pr_curve, generate_conf_matrix
import locpix.evaluate.metrics as metrics
from sklearn.metrics import precision_recall_curve, auc
import polars as pl
from datetime import datetime
import matplotlib.pyplot as plt
import pickle as pkl
from locpix.visualise import vis_img
import argparse
from locpix.scripts.img_seg import membrane_performance_config
import tkinter as tk
from tkinter import filedialog


def main():

    parser = argparse.ArgumentParser(description="Membrane performance metrics on data")
    parser.add_argument(
        "-i",
        "--project_directory",
        action="store",
        type=str,
        help="the location of the project directory",
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store",
        type=str,
        help="the location of the .yaml configuaration file\
                             for preprocessing",
    )

    args = parser.parse_args()

    # input project directory
    if args.project_directory is not None:
        project_folder = args.project_directory
    else:
        root = tk.Tk()
        root.withdraw()
        project_folder = filedialog.askdirectory(title="Project directory")

    if args.config is not None:
        # load yaml
        with open(args.config, "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            membrane_performance_config.parse_config(config)
    else:
        root = tk.Tk()
        root.withdraw()
        gt_file_path = filedialog.askdirectory(title="GT file path")
        config = membrane_performance_config.config_gui(gt_file_path)

    # list items
    gt_file_path = os.path.join(project_folder, "annotate/annotated")
    try:
        files = os.listdir(gt_file_path)
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    fig_train, ax_train = plt.subplots()
    fig_test, ax_test = plt.subplots()

    linestyles = ["dashdot", "-", "--"]
    methods = ["classic", "cellpose", "ilastik"]

    for index, method in enumerate(methods):

        print(f"{method} ...")

        # get folder names
        if method == "ilastik":
            seg_folder = os.path.join(
                project_folder, "ilastik/output/membrane/prob_map"
            )
        else:
            seg_folder = os.path.join(project_folder, f"{method}/membrane/prob_map")
        output_df_folder = os.path.join(
            project_folder, f"membrane_performance/{method}/membrane/seg_dataframes"
        )
        output_seg_imgs = os.path.join(
            project_folder, f"membrane_performance/{method}/membrane/seg_images"
        )
        output_train_pr = os.path.join(
            project_folder, f"membrane_performance/{method}/membrane/train_pr"
        )
        output_test_pr = os.path.join(
            project_folder, f"membrane_performance/{method}/membrane/test_pr"
        )
        output_metrics = os.path.join(
            project_folder, f"membrane_performance/{method}/membrane/metrics"
        )
        output_overlay_pr_curves = os.path.join(
            project_folder, "membrane_performance/overlaid_pr_curves"
        )
        output_conf_matrix = os.path.join(
            project_folder, f"membrane_performance/{method}/membrane/conf_matrix"
        )

        # create output folders
        folders = [
            output_df_folder,
            output_seg_imgs,
            output_train_pr,
            output_test_pr,
            output_metrics,
            output_overlay_pr_curves,
            output_conf_matrix,
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

            if file.removesuffix(".parquet") not in config["train_files"]:
                continue

            print("File ", file)

            # load df
            item = datastruc.item(None, None, None, None)
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
            item.load_from_parquet(os.path.join(gt_file_path, file))

            # load in histograms
            input_histo_folder = os.path.join(project_folder, "annotate/histos")
            histo_loc = os.path.join(input_histo_folder, item.name + ".pkl")
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
        metrics.aggregated_metrics(
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
    # ax_train.legend([handles_train[idx] for idx in order],
    #                   [methods[idx] for idx in order])
    # ax_test.legend([handles_test[idx] for idx in order],
    #                   [methods[idx] for idx in order])

    fig_train.savefig(os.path.join(output_overlay_pr_curves, "_train.png"), dpi=600)
    fig_test.savefig(os.path.join(output_overlay_pr_curves, "_test.png"), dpi=600)

    # save yaml file
    yaml_save_loc = os.path.join(project_folder, "membrane_performance.yaml")
    with open(yaml_save_loc, "w") as outfile:
        yaml.dump(config, outfile)


if __name__ == "__main__":
    main()
