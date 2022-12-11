"""Module for rendering point clouds of various forms"""

import numpy as np
import polars as pl


def visualise_seg_item(item, column):
    """Function for visualising the segmentation of
    a point cloud and saving
    Where column is the column the histogram values are
    in

    Note that as not every pixel will contain localisations
    there may be pixels which get set to zero.


    Args:
        item (datastruc.item) : This contains the
            df and the labels to visualise.
            This must be true!
        column (string) : Column containing the
            segmention labels

    """

    labels = item.df.select(pl.col(column)).to_numpy()
    x_pixels = item.df.select(pl.col("x_pixel")).to_numpy()
    y_pixels = item.df.select(pl.col("y_pixel")).to_numpy()

    histo_width = item.histo[0].shape[0]
    histo_height = item.histo[0].shape[1]

    histo = np.empty((histo_width, histo_height))

    histo[x_pixels, y_pixels] = labels

    return histo
