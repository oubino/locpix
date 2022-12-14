"""Module for calculating performance metrics"""

from locpix.preprocessing import datastruc
import os
import polars as pl
import numpy as np


def metric_calculation(item: datastruc.item, labels):
    """Calculate TP, TN, FP and FN for df for each label present

    Args:
        item (datastruc.item): Item containing the
            dataframe with gt and pred label
        labels (list) : List of labels to calculate results for

    Returns:
        dict : Results
            The key is the label and at each index
            is a dictionary containing TP, TN, FP, FN
            e.g. {0:{'TP': 0.6, 'FP':.3, ...}, 1:{'TP': 0.4, 'FP':.2, ...}]
            means we have two labels 0 and 1"""

    # visualise for checking
    # visualise(gt_df, pred_df, labels)

    results = {}

    # check no duplicates
    if len(item.df) != len(item.df.unique()):
        input(
            "You have duplicates in dataframe - remove this line and \
            replace with value error once appropriate"
        )

    for label in labels:
        tp_df = item.df.filter(
            (pl.col("gt_label") == label) & (pl.col("pred_label") == label)
        )
        TP = len(tp_df)
        fp_df = item.df.filter(
            (pl.col("gt_label") != label) & (pl.col("pred_label") == label)
        )
        FP = len(fp_df)
        tn_df = item.df.filter(
            (pl.col("gt_label") != label) & (pl.col("pred_label") != label)
        )
        TN = len(tn_df)
        fn_df = item.df.filter(
            (pl.col("gt_label") == label) & (pl.col("pred_label") != label)
        )
        FN = len(fn_df)

        results[label] = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

    return results


def mean_metrics(results, labels):
    """Calculate the mean metrics

    Args:
        results (dict) : Nested dictionary where each key
            is a label representing another dictionary
            containing the number of TP, TN, FP and FN
            e.g. results[2]["TP"] is number of true
            positives for label 2
        labels (list) : List of labels"""

    acc_list = []  # empty list will have length = number of labels
    iou_list = []
    recall_list = []
    for label in labels:
        TP = results[label]["TP"]
        FP = results[label]["FP"]
        TN = results[label]["TN"]
        FN = results[label]["FN"]
        acc_list.append((TP + TN) / (TP + TN + FP + FN))
        iou_list.append((TP) / (TP + FP + FN))
        recall_list.append(TP / (TP + FN))
    macc = np.mean(acc_list)
    miou = np.mean(iou_list)
    return iou_list, acc_list, recall_list, miou, macc


def aggregated_metrics(
    files_folder, save_loc, gt_label_map, add_metrics={}, metadata={}
):
    """Calculate metrics over files, where each
    file should represent a parquet file which is a
    datastruc.item

    Args:
        files_folder (string): Location in memory of the
        datastruc.item saved as a .parquet file
        save_loc (string): where to save the calculated
            metrics
        gt_label_map (dict): dictionary where each key
            is an integer and the value is the associated
            label
        metadata (dictionary): Any additional metadata
            to save"""

    # list items
    try:
        files = os.listdir(files_folder)
    except FileNotFoundError:
        raise ValueError("There should be some files to open")

    # create empty aggregated results dictionary, with each key
    # being a label
    agg_results = {}

    labels = gt_label_map.keys()
    for label in labels:
        agg_results[label] = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}

    # for each file want to aggregate TP, TN, FP, FN for each class
    for file in files:

        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(files_folder, file))

        file_results = metric_calculation(item, labels)

        # add file results to aggregated results
        for label in labels:
            agg_results[label]["TP"] += file_results[label]["TP"]
            agg_results[label]["TN"] += file_results[label]["TN"]
            agg_results[label]["FP"] += file_results[label]["FP"]
            agg_results[label]["FN"] += file_results[label]["FN"]

    # create empty results dictionary
    results = {}

    # calculate macc/miou/oacc
    iou_list, acc_list, recall_list, miou, macc = mean_metrics(agg_results, labels)
    results["iou_list"] = iou_list
    results["acc_list"] = acc_list
    results["recall_list"] = recall_list
    results["macc"] = macc
    results["miou"] = miou

    # calculate oa
    tp_running_total = 0
    fp_running_total = 0
    for label in labels:
        TP = agg_results[label]["TP"]
        FP = agg_results[label]["FP"]
        tp_running_total += TP
        fp_running_total += FP
    oacc = (tp_running_total) / (tp_running_total + fp_running_total)
    results["oacc"] = oacc

    # save to .csv in output results
    # df = pl.DataFrame(results)
    # df.write_csv(output_results)

    # sklearn to check and also speed
    pred_list = np.array([])
    gt_list = np.array([])

    for file in files:

        item = datastruc.item(None, None, None, None)
        item.load_from_parquet(os.path.join(files_folder, file))

        gt = item.df.select(pl.col("gt_label")).to_numpy()
        pred = item.df.select(pl.col("pred_label")).to_numpy()

        gt_list = np.append(gt_list, gt)
        pred_list = np.append(pred_list, pred)

    """
    print("Sklearn results")
    acc = sklearn.metrics.accuracy_score(gt_list, pred_list)
    report = sklearn.metrics.classification_report(gt_list, pred_list)
    confusion_matrix = sklearn.metrics.confusion_matrix(gt_list, pred_list)
    prec = sklearn.metrics.precision_score(gt_list, pred_list)
    recall = sklearn.metrics.recall_score(gt_list, pred_list)
    print("acc", acc)
    print("report", report)
    print("confusion matrix", confusion_matrix)
    print("prec", prec)
    print("recall", recall)
    """

    # Create text file and save overall results/configuration to this
    lines = ["Overall results", "-----------"]
    sections = [results, add_metrics, metadata]
    for section in sections:
        for key, value in section.items():
            lines.append(f"{key} : {value}")
    lines.append(f"gt label map:  {gt_label_map}")

    with open(save_loc, "w") as f:
        f.writelines("\n".join(lines))
