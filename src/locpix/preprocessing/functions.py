"""Preprocessing module.

This module contains functions to preprocess the data,
including (add as added):
- file to datastructure
    - convert .csv to datastructure
    - convert .parquet to datastructure
"""

import polars as pl
import os
from . import datastruc


def file_to_datastruc(
    input_file,
    file_type,
    dim,
    channel_col,
    frame_col,
    x_col,
    y_col,
    z_col,
    channel_choice=None,
    channel_label=None,
):
    """Loads in .csv or .parquet and converts to the required datastructure.

    Currently considers the following columns: channel frame x y z
    Also user can specify the channels they want to consider, these
    should be present in the channels column

    Args:
        input_file (string) : Location of the file
        file_type (string) : Either csv or parquet
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
        channel_label (list of strings) : If specified then this is the
            label for each channel i.e. ['egfr', 'ereg','unk'] means
            channel 0 is egfr protein, channel 1 is ereg proteins and
            channel 2 is unknown

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

    # check file type parquet or csv
    if file_type != "csv" and file_type != "parquet":
        raise ValueError(f"{file_type} is not supported, should be csv or parquet")

    # Load in data
    if dim == 2:
        if file_type == "csv":
            df = pl.read_csv(input_file, columns=[channel_col, frame_col, x_col, y_col])
        elif file_type == "parquet":
            df = pl.read_parquet(
                input_file, columns=[channel_col, frame_col, x_col, y_col]
            )
        df = df.rename(
            {channel_col: "channel", frame_col: "frame", x_col: "x", y_col: "y"}
        )
    elif dim == 3:
        if file_type == "csv":
            df = pl.read_csv(
                input_file, columns=[channel_col, frame_col, x_col, y_col, z_col]
            )
        elif file_type == "parquet":
            df = pl.read_parquet(
                input_file, columns=[channel_col, frame_col, x_col, y_col]
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
    if file_type == "csv":
        name = os.path.basename(os.path.normpath(input_file)).removesuffix(".csv")
    elif file_type == "parquet":
        name = os.path.basename(os.path.normpath(input_file)).removesuffix(".parquet")

    return datastruc.item(name, df, dim, channels, channel_label)
