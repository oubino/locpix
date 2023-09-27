"""Datastruc module.

This module contains definitions of the datastructure the
SMLM dataitem will be parsed as.

"""

import numpy as np
import napari
import matplotlib.pyplot as plt
import polars as pl
import pyarrow.parquet as pq
import ast
import os
import json
import warnings

_interpolate = {
    "log2": lambda d: np.log2(d),
    "log10": lambda d: np.log10(d),
    "linear": lambda d: d,
}


class item:
    """smlm datastructure.

    This is the basic datastructure which will contain all the information
    from a point set that is needed

    Attributes:
        name (string) : Contains the name of the item
        df (polars dataframe): Dataframe with the data contained, containing
            columns: 'channel' 'frame' 'x' 'y' 'z'.
            If manual annotation is done an additional column 'gt_label'
            will be present
        dim (int): Dimensions of data
        channels (list): list of ints, representing channels user wants
            to consider in the original data
        channel_label (list of strings) : The label for each channel
            i.e. ['egfr', 'ereg','unk'] means
            channel 0 is egfr protein, channel 1 is ereg proteins and
            channel 2 is unknown
        histo (dict): Dictionary of 2D or 3D arrays. Each key corresponds
            to the channel for which the histogram
            contains the relevant binned data, in form [X,Y,Z]
            i.e. histo[1] = histogram of channel 1 localisations.
            Note that if considering an image, need to transpose
            the histogram to follow image conventions.
        histo_edges (tuple of lists; each list contains floats):
            Tuple containing (x_edges,y_edges) or (x_edges, y_edges, z_edges)
            where x/y/z_edges are list of floats, each representing the
            edge of the bin in the original space.
            e.g. ([0,10,20],[13,25,20],[2,3,4])
        histo_mask (numpy array): Array containing integers where each should
            represent a different label of the MANUAL segmentation
            0 is reserved for background, is of form [X,Y,Z]
        bin_sizes (tuple of floats): Size of bins of the histogram
            e.g. (23.2, 34.5, 21.3)
        gt_label_map (dict): Dictionary with integer keys
            representing the gt labels for each localisation
            with value being a string, representing the
            real concept e.g. 0:'dog', 1:'cat'
    """

    def __init__(
        self,
        name,
        df,
        dim,
        channels,
        channel_label,
        histo={},
        histo_edges=None,
        histo_mask=None,
        bin_sizes=None,
        gt_label_map={},
    ):
        """Initialises item"""

        self.name = name
        self.df = df
        self.histo = histo
        self.histo_edges = histo_edges
        self.histo_mask = histo_mask
        self.dim = dim
        self.bin_sizes = bin_sizes
        self.channels = channels
        self.channel_label = channel_label
        self.gt_label_map = gt_label_map

        # channel labels and channel choice need to be same in length
        if (channels is not None) or (channel_label is not None):
            if len(channels) != len(channel_label):
                raise ValueError(
                    f"Labels for each channel is length {len(channel_label)}\n"
                    f"Channels is length {len(channels)}\n"
                    f"These must be the same length!"
                )

    # def save(self, save_loc):
    #     """Save the item

    #         Args:
    #             save_loc (string): Location to save the .pkl file"""

    #     dict = {"name": self.name, "df": self.df, "dim": self.dim,
    #             "channels": self.channels, "histo": self.histo,
    #             "histo_edges": self.histo_edges,
    #             "histo_mask": self.histo_mask,
    #             "bin_sizes": self.bin_sizes}
    #     pickle.dump(dict, open(save_loc, "wb"), pickle.HIGHEST_PROTOCOL)

    # def load(self, input_file):
    #     """ Loads item saved as .pkl file

    #         Args:
    #             input_file (string) : Location of the .pkl file to
    #                 load dataitem from"""

    #     with open(input_file, 'rb') as f:
    #         dict = pickle.load(f)
    #     self.__init__(name=dict['name'], df=dict['df'], dim=dict['dim'],
    #                   channels=dict['channels'],
    #                   histo=dict['histo'],
    #                   histo_edges=dict['histo_edges'],
    #                   histo_mask=dict['histo_mask'],
    #                   bin_sizes=dict['bin_sizes'])

    def chan_2_label(self, chan):
        """Returns the label associated with the channel specified

        Args:
            chan (int) : Integer representing the channel"""

        return self.channel_label[chan]

    def label_2_chan(self, label):
        """Returns the channel associated with the channel label
        specified

        Args:
            label (string) : String representing the label you want
                to find the channel for"""

        if label not in self.channel_label:
            raise ValueError(
                "The label specified is not present in" "the channel labels"
            )

        return self.channel_label.index(label)

    def coord_2_histo(
        self,
        histo_size,
        cmap=["Greens", "Reds", "Blues", "Purples"],
        vis_interpolation="linear",
    ):
        """Converts localisations into histogram of desired size,
        with option to plot the image (histo.T).
        Note the interpolation is only applied for visualisation,
        not for the actual data in the histogram!

        Args:
            histo_size (tuple): Tuple representing number of
                bins/pixels in x,y,z
            cmap (list of strings) : The colourmaps used to
                plot the histograms
            plot (bool): Whether to plot the output
            vis_interpolation (string): How to inerpolate
                the image for visualisation"""

        # get max and min x/y/(z) values
        df_max = self.df.max()
        df_min = self.df.min()

        if self.dim == 2:
            x_bins, y_bins = histo_size
            x_max = df_max["x"][0]
            y_max = df_max["y"][0]
            x_min = df_min["x"][0]
            y_min = df_min["y"][0]
        elif self.dim == 3:
            x_bins, y_bins, z_bins = histo_size
            z_max = df_max["z"][0]
            z_min = df_min["z"][0]

        # if instead want desired bin size e.g. 50nm, 50nm, 50nm
        # number of bins required for desired bin_size
        # note need to check if do this that agrees with np.digitize
        # and need to make sure that same issue we had before
        # with the last localisation is dealt with
        # x_bins = int((self.max['x'] - self.min['x']) / bin_size[0])
        # y_bins = int((self.max['y'] - self.min['y']) / bin_size[1])
        # z_bins = int((self.max['z'] - self.min['z']) / bin_size[2])

        # size of actual bins, given the number of bins (should be
        # very close to desired tests size)
        x_bin_size = (x_max - x_min) / x_bins
        y_bin_size = (y_max - y_min) / y_bins
        # need to increase bin size very marginally to include last localisation
        x_bin_size = x_bin_size * 1.001
        y_bin_size = y_bin_size * 1.001
        # location of edges of histogram, based on actual tests size
        x_edges = [x_min + x_bin_size * i for i in range(x_bins + 1)]
        y_edges = [y_min + y_bin_size * i for i in range(y_bins + 1)]
        # treat z separately, as often only in 2D
        if self.dim == 3:
            z_bin_size = (z_max - z_min) / z_bins
            # need to increase bin size very marginally to include last localisation
            z_bin_size = z_bin_size * 1.001
            z_edges = [z_min + z_bin_size * i for i in range(z_bins + 1)]

        # size per tests in nm; location of histo edges in original space
        if self.dim == 2:
            self.bin_sizes = (x_bin_size, y_bin_size)
            self.histo_edges = (x_edges, y_edges)
        if self.dim == 3:
            self.bin_sizes = (x_bin_size, y_bin_size, z_bin_size)
            self.histo_edges = (x_edges, y_edges, z_edges)

        print("-- Bin sizes -- ")
        print(self.bin_sizes)

        if self.dim == 2:
            # 2D histogram for every channel, assigned to self.histo (dict)
            for chan in self.channels:
                df = self.df.filter(pl.col("channel") == chan)
                sample = np.array((df["x"], df["y"]))
                sample = np.swapaxes(sample, 0, 1)
                # (D, N) where D is
                # self.dim and N is number of localisations
                self.histo[chan], _ = np.histogramdd(sample, bins=self.histo_edges)

        if self.dim == 3:
            # 3D histogram for every channel, assigned to self.histo (dict)
            for chan in self.channels:
                df = self.df[self.df["channel"] == chan]
                sample = np.array((df["x"], df["y"], df["z"]))
                sample = np.swapaxes(sample, 0, 1)
                # (D, N) where D is self.dim and N is number of
                # localisations
                self.histo[chan], _ = np.histogramdd(sample, bins=self.histo_edges)

        plt.close()

        # work out pixel for each localisations
        self._coord_2_pixel()

        # check digitize agree
        # df_min = self.df.min()
        # x_min = df_min["x_pixel"][0]
        # y_min = df_min["y_pixel"][0]

        # df_max = self.df.max()
        # x_max = df_max["x_pixel"][0]
        # y_max = df_max["y_pixel"][0]

        # print('check')
        # print(x_min, x_max)
        # print(y_min, y_max)

        # print('check')
        # for chan in self.channels:
        #    df = self.df.filter(pl.col("channel") == chan)
        #    my_x = df["x_pixel"].to_numpy()
        #    my_y = df["y_pixel"].to_numpy()
        #    their_x = np.digitize(df['x'], bins=self.histo_edges[0])
        #    their_y = np.digitize(df['y'], bins=self.histo_edges[1])
        #    print('assert equal')
        #    np.testing.assert_array_equal(my_x, their_x-1)
        #    np.testing.assert_array_equal(my_y, their_y-1)

    def _coord_2_pixel(self):
        """Calculate the pixels corresponding to each localisation"""

        # drop pixel columns if already present
        for col in ["x_pixel", "y_pixel", "z_pixel"]:
            if col in self.df.columns:
                self.df = self.df.drop(col)

        # necessary for pd.eval below
        df_min = self.df.min()
        x_min = df_min["x"][0]
        y_min = df_min["y"][0]

        if self.dim == 2:
            x_pixel_width, y_pixel_width = self.bin_sizes
        elif self.dim == 3:
            x_pixel_width, y_pixel_width, z_pixel_width = self.bin_sizes

        # calculate pixel indices for localisations
        self.df = self.df.select(
            [
                pl.all(),
                pl.col("x").map(lambda q: (q - x_min) / x_pixel_width).alias("x_pixel"),
                pl.col("y").map(lambda q: (q - y_min) / y_pixel_width).alias("y_pixel"),
            ]
        )
        # floor the pixel locations
        self.df = self.df.with_columns(pl.col("x_pixel").cast(int, strict=True))
        self.df = self.df.with_columns(pl.col("y_pixel").cast(int, strict=True))

        # localisations at the end get assigned to outside the histogram,
        # therefore need to be assigned to previous pixel
        # self.df = self.df.with_columns(
        #    pl.when(pl.col("x_pixel") == self.df.max()["x_pixel"][0])
        #    .then(self.df.max()["x_pixel"][0] - 1)
        #    .otherwise(pl.col("x_pixel"))
        #    .alias("x_pixel")
        # )
        # self.df = self.df.with_columns(
        #    pl.when(pl.col("y_pixel") == self.df.max()["y_pixel"][0])
        #    .then(self.df.max()["y_pixel"][0] - 1)
        #    .otherwise(pl.col("y_pixel"))
        #    .alias("y_pixel")
        # )

        if self.dim == 3:
            z_min = df_min["z"][0]
            # calculate pixel indices for localisations
            self.df = self.df.select(
                [
                    pl.all(),
                    pl.col("z")
                    .map(lambda q: (q - z_min) / z_pixel_width)
                    .alias("z_pixel"),
                ]
            )
            # floor the pixel locations
            self.df = self.df.with_columns(pl.col("z_pixel").cast(int, strict=True))
            # localisations at the end get assigned to outside the histogram,
            # therefore need to be assigned
            # to previous pixel
            # self.df = self.df.with_columns(
            #    pl.when(pl.col("z_pixel") == self.df.max()["z_pixel"][0])
            #    .then(self.df.max()["z_pixel"][0] - 1)
            #    .otherwise(pl.col("z_pixel"))
            #    .alias("z_pixel")
            # )

    def manual_segment(
        self,
        cmap=["green", "red", "blue", "bop purple"],
        relabel=False,
        markers_loc=None,
    ):
        """Manually segment the image (histogram.T). Return the segmented
        histogram and extra column in dataframe corresponding to label.
        0 should be reserved for background

        Args:
            cmap (list of strings) : Colourmaps napari uses to
                plot the histograms

        Returns:
            markers (list) : Coordinates of markers if added
        """

        # if already has gt label raise error
        if "gt_label" in self.df.columns and not relabel:
            raise ValueError(
                "Manual segment cannot be called on a file which\
                              already has gt labels in it"
            )

        while True:
            if self.dim == 2:
                # overlay all channels for src
                if len(self.channels) != 1:
                    # create the viewer and add each channel (first channel on own,
                    # then iterate through others)
                    colormap_list = cmap
                    # note image shape when plotted: [x, y]
                    viewer = napari.view_image(
                        self.histo[self.channels[0]].T,
                        name=f"Channel {self.channels[0]}/"
                        f"{self.chan_2_label(self.channels[0])}",
                        rgb=False,
                        blending="additive",
                        colormap=colormap_list[0],
                        gamma=2,
                        contrast_limits=[0, 30],
                    )
                    for index, chan in enumerate(self.channels[1:]):
                        viewer.add_image(
                            self.histo[chan].T,
                            name=f"Channel {chan}/{self.chan_2_label(chan)}",
                            rgb=False,
                            blending="additive",
                            colormap=colormap_list[index + 1],
                            gamma=2,
                            contrast_limits=[0, 30],
                        )
                    # add labels if present
                    if relabel:
                        # note this has to be called after coord_2_histo to be in the
                        # correct shape
                        histo_mask = self.render_seg()
                        viewer.add_labels(histo_mask.T, name="Labels")
                        if markers_loc is not None:
                            markers = np.load(markers_loc, allow_pickle=True)
                            if markers.any() is not None:
                                viewer.add_points(markers, name="Points")
                    napari.run()

                # only one channel
                else:
                    img = self.histo[self.channels[0]].T
                    # create the viewer and add the image
                    viewer = napari.view_image(
                        img,
                        name=f"Channel {self.channels[0]}/"
                        f"{self.chan_2_label(self.channels[0])}",
                        rgb=False,
                        gamma=2,
                        contrast_limits=[0, 30],
                    )
                    # add labels if present
                    if relabel:
                        warnings.warn("Haven't loaded in markers")
                        # note this has to be called after coord_2_histo to be in the
                        # correct shape
                        histo_mask = self.render_seg()
                        viewer.add_labels(histo_mask.T, name="Labels")
                        if markers_loc is not None:
                            markers = np.load(markers_loc, allow_pickle=True)
                            if markers.any() is not None:
                                viewer.add_points(markers, name="Points")

                    napari.run()

                try:
                    markers = viewer.layers["Points"].data
                    x = [[int(float(j)) for j in i] for i in markers]
                    markers = [tuple(i) for i in x]
                except KeyError:
                    print("You must add points!")
                    markers = None

                # histogram mask should be assigned to GUI output
                try:
                    self.histo_mask = viewer.layers["Labels"].data.T
                    break
                except KeyError:
                    print("You must add labels!")

            elif self.dim == 3:
                print("segment 3D image")

        # segment the coordinates
        self._manual_seg_pixel_2_coord()

        return markers

    def _manual_seg_pixel_2_coord(self):
        """Get the localisations associated with manual annotation.
        Each integer should represent a different label, where 0 is reserved
        for background.
        """

        if self.dim == 2:

            # create dataframe
            flatten_mask = np.ravel(self.histo_mask)
            mesh_grid = np.meshgrid(
                range(self.histo_mask.shape[0]), range(self.histo_mask.shape[1])
            )
            x_pixel = np.ravel(mesh_grid[1])
            y_pixel = np.ravel(mesh_grid[0])
            label = flatten_mask
            data = {"x_pixel": x_pixel, "y_pixel": y_pixel, "gt_label": label}
            mask_df = pl.DataFrame(
                data,
                schema=[
                    ("x_pixel", pl.Int64),
                    ("y_pixel", pl.Int64),
                    ("gt_label", pl.Int64),
                ],
            ).sort(["x_pixel", "y_pixel"])

            # drop gt_label if already present
            if "gt_label" in self.df.columns:
                self.df = self.df.drop("gt_label")

            # join mask dataframe
            self.df = self.df.join(mask_df, how="inner", on=["x_pixel", "y_pixel"])

            # sanity check
            # print(len(self.df))
            # print(self.df.columns)
            # print(self.df.head(10))

        elif self.dim == 3:
            print("segment the 3d coords")

    def mask_pixel_2_coord(self, img_mask: np.ndarray) -> pl.DataFrame:
        """For a given mask over the image (value at each pixel
        normally representing a label), return the dataframe with a column
        giving the value for each localisation. Note that it is
        assumed that the img_mask is a mask of the image,
        therefore have to transpose img_mask for it to be in the same
        configuration as the histogram

        Note we also use this for  labels and when
        the img_mask represents probabilities.

        Args:
            img_mask (np.ndarray): Mask over the image -
            to reiterate, to convert this to histogram space need
            to transpose it

        Returns:
            df (polars dataframe): Original dataframe with
            additional column with the predicted label"""

        if self.dim == 2:
            # list of mask dataframes, each mask dataframe contains
            # (x,y,label) columns
            # transpose the image mask to histogram space
            histo_mask = img_mask.T

            # create dataframe
            flatten_mask = np.ravel(histo_mask)
            mesh_grid = np.meshgrid(
                range(histo_mask.shape[0]), range(histo_mask.shape[1])
            )
            x_pixel = np.ravel(mesh_grid[1])
            y_pixel = np.ravel(mesh_grid[0])
            label = flatten_mask
            data = {"x_pixel": x_pixel, "y_pixel": y_pixel, "pred_label": label}
            mask_df = pl.DataFrame(
                data,
                schema=[
                    ("x_pixel", pl.Int64),
                    ("y_pixel", pl.Int64),
                    ("pred_label", pl.Float64),
                ],
            ).sort(["x_pixel", "y_pixel"])

            # sanity check
            # print(len(self.df))
            # print(self.df.columns)
            # print(self.df.head(10))

            # join mask dataframe
            df = self.df.join(mask_df, how="inner", on=["x_pixel", "y_pixel"])

            # sanity check
            # print(len(df))
            # print(df.columns)
            # print(df.head(10))

            return df

        elif self.dim == 3:
            print("segment the 3d coords")

    def save_df_to_csv(
        self, csv_loc, drop_zero_label=False, drop_pixel_col=True, save_chan_label=True
    ):
        """Save the dataframe to a .csv with option to:
                drop positions which are background
                drop the column containing pixel information
                save additional column with labels for each
                    localisation

        Args:
            csv_loc (String): Save the csv to this location
            drop_zero_label (bool): If True then only non zero
                label positions are saved to csv
            drop_pixel_col (bool): If True then don't save
                the column with x,y,z pixel
            save_chan_label (bool) : If True then save an
                additional column for each localisation
                containing the label for each channel

        Returns:
            None"""

        save_df = self.df

        if drop_pixel_col:
            # don't want to save x,y pixel to csv
            save_df = save_df.drop("x_pixel")
            save_df = save_df.drop("y_pixel")
            if self.dim == 3:
                save_df = save_df.drop("z_pixel")

        # rearrange so x,y,z, ...,labels,channels
        # if self.dim == 2:
        #    cols = ['x', 'y', 'gt_label', 'channel']
        # elif self.dim == 3:
        #    cols = ['x', 'y', 'z', 'gt_label', 'channel']
        # save_df_cols = save_df.columns
        # cols = [col for col in cols if col in save_df_cols] +
        # [col for col in save_df_cols if col #]not in cols]
        # save_df = save_df[cols]

        # drop rows with zero label
        if drop_zero_label:
            save_df = save_df.filter(pl.col("gt_label") != 0)

        # save channel label as well
        if save_chan_label:
            label_df = pl.DataFrame({"chan_label": self.channel_label}).with_row_count(
                "channel"
            )
            label_df = label_df.with_columns(pl.col("channel").cast(pl.Int64))
            save_df = save_df.join(label_df, on="channel", how="inner")

        # save to location
        save_df.write_csv(csv_loc, sep=",")

    def save_to_parquet(
        self,
        save_folder,
        drop_zero_label=False,
        drop_pixel_col=False,
        gt_label_map=None,
        overwrite=False,
    ):
        """Save the dataframe to a parquet with option to drop positions which
           are background and can drop the column containing pixel
           information

        Args:
            save_folder (String): Save the df to this folder
            drop_zero_label (bool): If True then only non zero
                label positions are saved to parquet
            drop_pixel_col (bool): If True then don't save
                the column with x,y,z pixel
            gt_label_map (dict): Dictionary with integer keys
                representing the gt labels for each localisation
                with value being a string, representing the
                real concept e.g. 0:'dog', 1:'cat'
            overwrite (bool): Whether to overwrite

        Returns:
            None
        """

        save_df = self.df

        if drop_pixel_col:
            # don't want to save x,y pixel to csv
            save_df = save_df.drop("x_pixel")
            save_df = save_df.drop("y_pixel")
            if self.dim == 3:
                save_df = save_df.drop("z_pixel")

        # drop rows with zero label
        if drop_zero_label:
            save_df = save_df.filter(pl.col("gt_label") != 0)

        # convert to arrow + add in metadata if doesn't exist
        arrow_table = save_df.to_arrow()

        # convert gt label map to bytes
        old_metadata = arrow_table.schema.metadata

        # convert to bytes
        gt_label_map = json.dumps(gt_label_map).encode("utf-8")
        meta_data = {
            "name": self.name,
            "dim": str(self.dim),
            "channels": str(self.channels),
            "channel_label": str(self.channel_label),
            "gt_label_map": gt_label_map,
            "bin_sizes": str(self.bin_sizes),
        }

        # add in label mapping
        # meta_data.update(gt_label_map)
        # merge existing with new meta data
        merged_metadata = {**meta_data, **(old_metadata or {})}
        arrow_table = arrow_table.replace_schema_metadata(merged_metadata)
        save_loc = os.path.join(
            save_folder, self.name + ".parquet"
        )  # note if change this need to adjust annotate.py
        if os.path.exists(save_loc) and not overwrite:
            raise ValueError(
                "Cannot overwite. If you want to overwrite please set overwrite==True"
            )
        pq.write_table(arrow_table, save_loc)

        # To access metadata write
        # parquet_table = pq.read_table(file_path)
        # parquet_table.schema.metadata ==> metadata
        # note if accessing keys need
        # parquet_table.schema.metadata[b'key_name'])
        # note that b is bytes

    def load_from_parquet(self, input_file):
        """Loads item saved as .parquet file

        Args:
            input_file (string) : Location of the .parquet file to
                load dataitem from"""

        # read in parquet file
        arrow_table = pq.read_table(input_file)

        # print("loaded metadata", arrow_table.schema.metadata)

        # metadata
        name = arrow_table.schema.metadata[b"name"].decode("utf-8")
        gt_label_map = json.loads(
            arrow_table.schema.metadata[b"gt_label_map"].decode("utf-8")
        )
        if gt_label_map is not None:
            # convert string keys to int keys for the mapping
            gt_label_map = {int(key): value for key, value in gt_label_map.items()}
        # convert string keys to int keys for the mapping
        dim = arrow_table.schema.metadata[b"dim"]
        dim = int(dim)
        channels = arrow_table.schema.metadata[b"channels"]
        channels = ast.literal_eval(channels.decode("utf-8"))
        channel_label = arrow_table.schema.metadata[b"channel_label"]
        channel_label = ast.literal_eval(channel_label.decode("utf-8"))
        bin_sizes = arrow_table.schema.metadata[b"bin_sizes"]
        bin_sizes = ast.literal_eval(bin_sizes.decode("utf-8"))
        df = pl.from_arrow(arrow_table)

        # print("channel label", channel_label)

        self.__init__(
            name=name,
            df=df,
            dim=dim,
            channels=channels,
            channel_label=channel_label,
            gt_label_map=gt_label_map,
            bin_sizes=bin_sizes,
        )

    def render_histo(self, labels=None):
        """Render the histogram from the .parquet file

        If labels are specified then the histogram is rendered in the order
        of these lables
        If not specified defaults to rendering in the channels specified
        in order by user e.g. [0,3,1,2]
        Assumes localisations have associated x_pixel and y_pixel already.

        Args:
            labels (list) : Order of labels to stack histograms in
                e.g. labels=['egfr','ereg'] means all images will
                be returned with egfr in channel 0 and ereg in
                channel 1

        Returns:
            histo (np.histogram) : Histogram of the localisation data
            channel_map (list) : List where the first value is the
                channel in the first axis of the histogram, second value
                is the channel in the second axis of the histogram etc.
                e.g. [1,3] : 1st channel is in 1st axis, 3rd channel in 2nd axis
            label_map (list) : List where the first value is the
                label in the first axis of the histogram, second value
                is the channel in the second axis of the histogram etc.
                e.g. ['egfr','ereg'] : egfr is in 1st axis, ereg in 2nd axis
        """

        histos = []

        df_max = self.df.max()
        x_bins = df_max["x_pixel"][0] + 1
        y_bins = df_max["y_pixel"][0] + 1

        if labels is None:
            channels = self.channels
            label_map = [self.chan_2_label(chan) for chan in channels]

        else:
            channels = [self.label_2_chan(label) for label in labels]
            label_map = labels

        for chan in channels:
            df = self.df.filter(pl.col("channel") == chan)

            histo = np.zeros((x_bins, y_bins))
            df = df.groupby(by=["x_pixel", "y_pixel"]).count()
            x_pixels = df["x_pixel"].to_numpy()
            y_pixels = df["y_pixel"].to_numpy()
            counts = df["count"].to_numpy()
            histo[x_pixels, y_pixels] = counts

            histos.append(histo)

        histo = np.stack(histos)
        channel_map = channels

        return histo, channel_map, label_map

    def render_seg(self):
        """Render the segmentation of the histogram"""

        labels = self.df.select(pl.col("gt_label")).to_numpy()
        x_pixels = self.df.select(pl.col("x_pixel")).to_numpy()
        y_pixels = self.df.select(pl.col("y_pixel")).to_numpy()

        histo_width = np.max(x_pixels) + 1
        histo_height = np.max(y_pixels) + 1

        histo = np.zeros((histo_width, histo_height), dtype=np.int64)

        histo[x_pixels, y_pixels] = labels

        return histo
