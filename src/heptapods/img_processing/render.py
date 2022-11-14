"""Module for rendering point clouds of various forms"""

from heptapods.preprocessing import datastruc
import numpy as np
import polars as pl


def visualise_seg_item(item, column):
    """Function for visualising the segmentation of
    a point cloud and saving
    Where column is the column the histogram values are 
    in

    Args:
        item (datastruc.item) : This contains the
            df and the labels to visualise.
            This must be true!

    """

    gt_labels = item.df.select(pl.col(column)).to_numpy()
    x_pixels = item.df.select(pl.col("x_pixel")).to_numpy()
    y_pixels = item.df.select(pl.col("y_pixel")).to_numpy()

    histo_width = item.histo[0].shape[0]
    histo_height = item.histo[0].shape[1]

    histo = np.empty((histo_width, histo_height))

    histo[x_pixels, y_pixels] = gt_labels

    return histo
