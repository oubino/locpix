""" Visualise module for images

Contains functions for visualising histograms
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import networkx as nx
from skimage.future.graph import RAG
from skimage.transform import resize

_interpolate = {
    "log2": lambda d: np.log2(d),
    "log10": lambda d: np.log10(d),
    "linear": lambda d: d,
}


def manual_threshold(array, threshold, how="linear") -> np.ndarray:
    """Perform manual threshold of the input array

    Args:
        array (np.ndarray): numpy array
        threshold (float or int): Threshold above
            which array will be thresholded
        how (string): How to interpolate the array

    Returns:
        thresh_array (np.ndarray): The thresholded output on the input array"""

    # interpolate the image
    array = _interpolate[how](array)
    array = np.clip(array, 0, None)  # clip to at least 0
    return np.where(array > threshold, array, 0)


def label_2_4_colours(labels: np.ndarray):
    """This takes in an array of labels
    e.g. [5,0,2,2,17,1,...]
    and returns a list where each label is now
    in range 0-3 such that no 2 integers next to
    each other are the same.
    This can be used for to plot so no 2 colours
    next to each other are the same

    Args:
        labels (np.ndarray): Array of ints
        each representing a unique label

    Returns:
        Array of ints each representing a
        unique label (between 0 and 3) no
        2 of same label should be next to each other"""

    graph = RAG(labels)
    d = nx.coloring.greedy_color(graph)
    return np.vectorize(d.get)(labels)


def img_2_grey(img: np.ndarray) -> np.ndarray:
    """converts img to 0 255 greyscale image

    Args:
        img (np.ndarray) : Image to be converted to greyscale

    Returns:
        img (np.ndarray): Image in uint8 form greyscale"""

    mini = np.min(img)
    maxi = np.max(img)
    img = ((img - mini) / (maxi - mini)) * 255
    return img.astype("uint8")


def visualise_seg(
    image_dict,
    segmentation,
    bin_sizes,
    channels,
    threshold=0,
    how="linear",
    alphas=[1, 0.5, 0.2, 0.1],
    alpha_seg=0.8,
    blend_overlays=False,
    cmap_img=None,
    cmap_seg=["m", "g", "lightsalmon", "r", "b"],
    figsize=(10, 10),
    origin="upper",
    save=False,
    save_loc=None,
    four_colour=True,
    background_one_colour=False,
):
    """Take in image and the associated segmentation and plot it,
    with option to convert to 4 colours and also option to choose to
    plot background as one particular colour
    (only relevant if four colour is true)

    Args:
        image_dict (dict) : Dictionary where each key is
            the channel for the respective image (which is
            the transpose of the histogram) i.e. img_dict[0]
            should be a img (histogram.T) for the 0th channel
        segmentation (np.ndarray) : Array of integers where
            each represents unique label in the segmentation
        bin_sizes (list) : List of sizes of the bins of the
            histogram
        channels (list): List of channels to plot
        threshold (int) : Threshold applied to images
            when plotting
        how (string) : Interpolation applied to image
            when plotting
        alphas (list) : List of alpha to be used in plt.imshow
        alpha_seg (float) : Value of alpha for segmentation
        blend_overlays (bool) : Whether to blend img and seg
        cmap_img (list) : List of cmaps to plot images
            (transpose of histograms)
        cmap_seg (list) : List of cmaps to plot segmentations
        figsize (tuple) : Tuple of figure size in 2D (x,y) sizes
        origin (string) : Location of origin in plot, image
            convention is 'upper' but for cartesian use 'lower'
        save (bool) : Whether to save
        save_loc (string) : Where to save if chosen to save
        four_colour (bool) : Whether to convert localisations
            to 4 colour
        background_one_colour (bool) : Whether to keep background
            as all same colour when doing 4 colour conversion

    """

    if cmap_img is None:
        cmap_img = [
            LinearSegmentedColormap.from_list("", ["k", "c"], N=256),
            LinearSegmentedColormap.from_list("", ["k", "y"], N=256),
            LinearSegmentedColormap.from_list("", ["k", "b"], N=256),
            LinearSegmentedColormap.from_list("", ["k", "g"], N=256),
        ]
    else:
        cmap_img = [
            LinearSegmentedColormap.from_list("", ["k", i], N=256) for i in cmap_img
        ]

    if four_colour:
        if background_one_colour:
            four_colour_seg = label_2_4_colours(segmentation)
            # now set all regions which were 0 before to 0
            # and increase four colour by 1
            # and add white to cmap_seg --> ensures
            # background comes up as untouched
            segmentation = np.where(segmentation != 0, four_colour_seg + 1, 0)
            cmap_seg = ["w"] + cmap_seg
        else:
            segmentation = label_2_4_colours(segmentation)

    fig, ax = plt.subplots(figsize=figsize)
    cmap_seg = ListedColormap(cmap_seg)

    # patches = []  # legend creation
    # plot for each channel in sequence
    for index, chan in enumerate(channels):
        img = manual_threshold(image_dict[chan], threshold=threshold, how=how)
        img = img_2_grey(img)
        img = np.where(img > 100, 255, 0)

        # resize img according to bin size - so that pixel size same in x and y
        h_old, w_old = img.shape
        img = resize(img, (h_old * bin_sizes[1] / bin_sizes[0], w_old))
        segmentation = resize(
            segmentation, (h_old * bin_sizes[1] / bin_sizes[0], w_old)
        )
        img = img[0:h_old, 0:w_old]
        segmentation = segmentation[0:h_old, 0:w_old]

        ax.imshow(
            img, cmap=cmap_img[index], origin=origin, alpha=alphas[index]
        )  # cmap=cmap_list[index]
        # x,y scale bars
        x = [15, 115]
        y = [465, 465]
        ax.plot(x, y, color="w", lw=3)
        ax.text(
            35,
            490,
            f"{bin_sizes[0]*100/1000:.0f}\u03BCm",
            fontsize=25,
            color="w",
            bbox={"facecolor": "black", "edgecolor": "none", "pad": 2, "alpha": 0.7},
        )
        ax.set_axis_off()
        # cmap = plt.cm.get_cmap(cmap_list[index]) # legend creation

        # legend creation
        # cmap = plt.cm.get_cmap(cmap_img[index])
        # patches.append(mpatches.Patch(color=cmap(255), label=f"Chan {chan}"))
        # plt.legend(
        #    handles=patches,
        #    bbox_to_anchor=(0.8, 1),
        #    loc=2,
        #    borderaxespad=0.0,
        #    prop={"size": 15},
        # )

    if blend_overlays:
        alphas = alpha_seg
    elif blend_overlays is False:
        alphas = np.where(segmentation > 0, alpha_seg, 0).astype(float)
    plt.imshow(segmentation, cmap=cmap_seg, alpha=alphas, origin=origin)

    if save:
        if save_loc is None:
            raise ValueError("Need a save location!")
        plt.savefig(save_loc, bbox_inches="tight")

    plt.close()
