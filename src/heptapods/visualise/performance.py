"""Various functions for visualising performance"""

import matplotlib.pyplot as plt


def plot_pr_curve(precision, recall, baseline, save_loc):
    """Plots the precision recall curve and saves

    Args:
        precision (list): List of precisions
        recall (list): List of recalls
        baseline (float): The baseline performance
            value

    Returns:
        None
    """

    plt.figure()
    lw = 6
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=lw,
    )

    # baseline is just always predicting all positive class
    plt.plot([0, 1], [baseline, baseline], color="navy", lw=lw, linestyle="--")
    plt.xlabel("Recall", fontsize=18)
    plt.ylabel("Precision", fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # this turns off the y ticks
    # plt.yticks([], [])
    plt.ylim((0, 1.1))
    plt.legend(loc="upper right")
    # tight layout
    plt.tight_layout()
    plt.savefig(save_loc, dpi=600)
