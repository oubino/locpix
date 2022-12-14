"""Preprocessing module.

This module contains functions to preprocess the data,
including (add as added):
- convert .csv to datastructure
"""

import polars as pl
import os
from . import datastruc


def csv_to_datastruc(
    input_file, dim, channel_col, frame_col, x_col, y_col, z_col, channel_choice=None
):
    """Loads in .csv and converts to the required datastructure.
    
    Currently considers the following columns: channel frame x y z
    Also user can specify the channels they want to consider, these
    should be present in the channels column

    Args:
        input_file (string) : Location of the .csv
        save_loc (string) : Location to save datastructure to
        dim (int) : Dimensions to consider either 2 or 3
        channel_col (string) : Name of column which gives channel
            for localisation
        frame_col (string) : Name of column which gives frame for localisation
        x_col (string) : Name of column which gives x for localisation
        y_col (string) : Name of column which gives y for localisation
        z_col (string) : Name of column which gives z for localisation
        channel_choice (list of ints) : If specified then this will be list
            of integers representing channels to be considered

    Returns:
        datastruc (SMLM_datastruc) : Datastructure containg the data
    """

    # Check dimensions correctly specified
    if dim != 2 and dim != 3:
        raise ValueError("Dimensions must be 2 or 3")
    if dim == 2 and z_col:
        raise ValueError("If dimensions are two no z should be specified")
    if dim == 3 and not z_col:
        raise ValueError("If dimensions are 3 then z_col must be specified")

    # Load in data
    if dim == 2:
        df = pl.read_csv(input_file, columns=[channel_col, frame_col, x_col, y_col])
        df = df.rename(
            {channel_col: "channel", frame_col: "frame", x_col: "x", y_col: "y"}
        )
    elif dim == 3:
        df = pl.read_csv(
            input_file, columns=[channel_col, frame_col, x_col, y_col, z_col]
        )
        df = df.rename(
            {
                channel_col: "channel",
                frame_col: "frame",
                x_col: "x",
                y_col: "y",
                z_col: "z",
            }
        )

    # Specify channels to consider
    # if channel_choice is None:
    #    channels = df["channel"].unique()
    #    channels = sorted(channels)
    # else:
    channels = channel_choice

    # Get name of file - assumes last part of input file name
    name = os.path.basename(os.path.normpath(input_file)).strip(".csv")

    return datastruc.item(name, df, dim, channels)
