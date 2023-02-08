"""Various functions for visualising performance"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_pr_curve(
    ax,
    label,
    linestyle,
    colour,
    precision,
    recall,
    baseline,
):
    """Plots the precision recall curve and saves

    Args:
        ax (matplotlib figure): Figure
        label (string): Label for the plot
        linestyle (string): Linestyle for the plot
        colour (string): Colour for the plot
        precision (list): List of precisions
        recall (list): List of recalls
        baseline (float): The baseline performance
            value
        pickle (bool): Whether to save the figure as pickle
            so can reload and replot

    Returns:
        None
    """

    lw = 2
    ax.plot(
        recall,
        precision,
        color=colour,
        lw=lw,
        linestyle=linestyle,
        label=label,
    )

    # baseline is just always predicting all positive class
    ax.plot([0, 1], [baseline, baseline], color="navy", lw=lw, linestyle="--")
    ax.plot([0, 1], [1, 1], color="#D41159", lw=lw, linestyle="-")
    ax.plot([1, 1], [0, 1], color="#D41159", lw=lw, linestyle="-")
    ax.set_xlabel("Recall", fontsize=18)
    ax.set_ylabel("Precision", fontsize=18)
    ax.set_xticks(np.linspace(0, 1, 11), fontsize=18)
    ax.set_yticks(np.linspace(0, 1, 11), fontsize=18)
    # this turns off the y ticks
    # plt.yticks([], [])
    ax.set_ylim((0, 1.1))
    ax.legend(loc="upper right")


def generate_binary_conf_matrix(tn, fp, fn, tp, classes, saveloc):
    """Generates a confusion matrix and saves it

    Args:
        tn (int) : Number of true negatives
        fp (int) : Number of false positives
        fn (int) : Number of false negatives
        tp (int) : Number of true positives
        classes (list): List of classes associated
            with each label - assumes labels are ordered
            0, 1 and that 0 is negative and 1 is positive
        saveloc (string): Location to save the confusion matrix
            to

    Returns:
        None
    """
    # conf_mat = confusion_matrix(gtlist, predlist)
    conf_mat = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(conf_mat, cmap="YlGn")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    plt.xlabel("Predicted class", labelpad=10)
    plt.ylabel("True class", labelpad=10)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, cmap="YlGn")
    cbar.ax.set_ylabel("", rotation=-90, va="bottom")

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_yticklabels(),
        rotation=90,
        ha="center",
        va="bottom",
        rotation_mode="anchor",
    )

    # Loop over data dimensions and create text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                "{:.2e}".format(conf_mat[i, j]),
                ha="center",
                va="center",
                color="r",
                size=16,
            )

    print("Check this")
    print(conf_mat)

    # ax.set_title("Confusion matrix")
    fig.tight_layout()

    # Per-class accuracy
    class_accuracy = 100 * conf_mat.diagonal() / conf_mat.sum(1)
    print("Class" + " | " + "Class Accuracy")
    for i in range(len(classes)):
        print(classes[i] + " | " + str(class_accuracy[i]))

    # save figure
    plt.savefig(saveloc, dpi=600)
