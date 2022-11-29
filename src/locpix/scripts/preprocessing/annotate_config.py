"""Annotate config

Module which defines functions for managing the config
files for annotate.
Including:

Creating a GUI for user to choose configuration
Parsing config file to check all specified"""

from PyQt5.QtWidgets import (
    QLabel,
    QApplication,
    QWidget,
    QHBoxLayout,
    QMessageBox,
    QLineEdit,
    QFormLayout,
    QCheckBox,
    QListWidget,
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator

default_config_keys = [
    "x_bins",
    "y_bins",
    "z_bins",
    "dim",
    "plot",
    "vis_interpolation",
    "input_folder",
    "histo_folder",
    "output_folder",
    "drop_zero_label",
    "gt_label_map",
    "save_img",
    "save_threshold",
    "save_interpolate",
    "background_one_colour",
    "four_colour",
    "alphas",
    "alpha_seg",
    "cmap_seg",
    "fig_size",
    "output_seg_folder",
    "vis_channels",
    "yaml_save_loc",
]


class InputWidget(QWidget):
    """Input
    Widget to take in user configuration
    Child of QWidget.
    Args:
        None
    Attributes:

    """

    def __init__(self, config):
        """Constructor
        Args:
            files (list) : List of files that will be preprocessed
            config (dict) : Dictionary containing the configuration
        """

        super().__init__()
        self.flo = QFormLayout()

        self.x_bins = QLineEdit("500")
        self.x_bins.setValidator(QIntValidator())
        self.x_bins.setToolTip("Number of bins in x dimension")
        self.flo.addRow("X bins", self.x_bins)

        self.y_bins = QLineEdit("500")
        self.y_bins.setValidator(QIntValidator())
        self.y_bins.setToolTip("Number of bins in y dimension")
        self.flo.addRow("Y bins", self.y_bins)

        self.z_bins = QLineEdit("500")
        self.z_bins.setValidator(QIntValidator())
        self.z_bins.setToolTip("Number of bins in z dimension")
        self.flo.addRow("Z bins", self.z_bins)

        self.dim = QLineEdit("2")
        self.dim.setValidator(QIntValidator())
        self.dim.setToolTip("Dimensions of the data either 2 or 3")
        self.flo.addRow("Dimensions", self.dim)

        self.plot = QCheckBox()
        self.plot.setChecked(True)
        self.plot.setToolTip("Whether plot should occur to screen")
        self.flo.addRow("Plot", self.plot)

        self.vis_interpolation = QListWidget()
        self.vis_interpolation.insertItem(0, "log2")
        self.vis_interpolation.insertItem(1, "log10")
        self.vis_interpolation.insertItem(2, "linear")
        self.vis_interpolation.item(0).setSelected(True)
        self.vis_interpolation.setToolTip(
            "Interpolation applied to the histogram when visualising the image of the histogram"
        )
        self.flo.addRow("Vis interpolation", self.vis_interpolation)

        self.input_folder = QLineEdit("output/preprocess/no_gt_label")
        self.input_folder.setToolTip("Input folder")
        self.flo.addRow("Input folder", self.input_folder)

        self.histo_folder = QLineEdit("output/annotate/histos")
        self.histo_folder.setToolTip("Output folder for histograms")
        self.flo.addRow("Histo folder", self.histo_folder)

        self.output_folder = QLineEdit("output/annotate/annotated")
        self.output_folder.setToolTip("Output folder for annotated")
        self.flo.addRow("Output folder", self.output_folder)

        self.drop_zero_label = QCheckBox()
        help = (
            "When saving annotations - you can choose to"
            "not save the localisations associated with the background"
            "which are assigned a label of zero"
        )
        self.drop_zero_label.setToolTip(help)
        self.flo.addRow("Drop zero label", self.drop_zero_label)

        h_box = QHBoxLayout()
        zero_label = QLabel("0: ")
        self.gt_label_map_zero = QLineEdit("background")
        h_box.addWidget(zero_label)
        h_box.addWidget(self.gt_label_map_zero)
        one_label = QLabel("1: ")
        self.gt_label_map_one = QLineEdit("membrane")
        h_box.addWidget(one_label)
        h_box.addWidget(self.gt_label_map_one)
        self.gt_label_map_zero.setToolTip("0 label in real terms")
        self.gt_label_map_one.setToolTip("1 lable in real terms")
        self.flo.addRow("Gt label map", h_box)

        self.save_img = QCheckBox()
        self.save_img.setChecked(True)
        self.save_img.setToolTip("Save segmentation images settings")
        self.flo.addRow("Save image", self.save_img)

        self.save_threshold = QLineEdit("0")
        self.save_threshold.setValidator(QDoubleValidator())
        self.flo.addRow("Save threshold", self.save_threshold)

        self.save_interpolation = QListWidget()
        self.save_interpolation.insertItem(0, "log2")
        self.save_interpolation.insertItem(1, "log10")
        self.save_interpolation.insertItem(2, "linear")
        self.save_interpolation.item(0).setSelected(True)
        self.flo.addRow("Save interpolation", self.save_interpolation)

        self.background_one_colour = QCheckBox()
        self.background_one_colour.setChecked(True)
        self.flo.addRow("Background one colour", self.background_one_colour)

        self.four_colour = QCheckBox()
        self.four_colour.setChecked(True)
        self.flo.addRow("Four colour", self.four_colour)

        h_box = QHBoxLayout()
        zero_alpha = QLabel("Alpha chan zero: ")
        self.alpha_zero = QLineEdit("1")
        self.alpha_zero.setValidator(QDoubleValidator())
        h_box.addWidget(zero_alpha)
        h_box.addWidget(self.alpha_zero)
        one_alpha = QLabel("Alpha chan one: ")
        self.alpha_one = QLineEdit(".5")
        self.alpha_one.setValidator(QDoubleValidator())
        h_box.addWidget(one_alpha)
        h_box.addWidget(self.alpha_one)
        two_alpha = QLabel("Alpha chan two: ")
        self.alpha_two = QLineEdit(".2")
        self.alpha_two.setValidator(QDoubleValidator())
        h_box.addWidget(two_alpha)
        h_box.addWidget(self.alpha_two)
        three_alpha = QLabel("Alpha chan three: ")
        self.alpha_three = QLineEdit(".1")
        self.alpha_three.setValidator(QDoubleValidator())
        h_box.addWidget(three_alpha)
        h_box.addWidget(self.alpha_three)
        self.flo.addRow("Alphas", h_box)

        self.alpha_seg = QLineEdit("0.8")
        self.alpha_seg.setValidator(QDoubleValidator())
        self.flo.addRow("Alpha seg", self.alpha_seg)

        h_box = QHBoxLayout()
        zero_cmap = QLabel("Zero cmap: ")
        self.zero_cmap = QLineEdit("k")
        h_box.addWidget(zero_cmap)
        h_box.addWidget(self.zero_cmap)
        one_cmap = QLabel("One cmap: ")
        self.one_cmap = QLineEdit("y")
        h_box.addWidget(one_cmap)
        h_box.addWidget(self.one_cmap)
        self.flo.addRow("Cmap seg", h_box)

        h_box = QHBoxLayout()
        fig_size_x = QLabel("Fig size x: ")
        self.fig_size_x = QLineEdit("10")
        self.fig_size_x.setValidator(QIntValidator())
        h_box.addWidget(fig_size_x)
        h_box.addWidget(self.fig_size_x)
        fig_size_y = QLabel("Fig size y: ")
        self.fig_size_y = QLineEdit("10")
        self.fig_size_y.setValidator(QIntValidator())
        h_box.addWidget(fig_size_y)
        h_box.addWidget(self.fig_size_y)
        self.flo.addRow("Fig size", h_box)

        self.output_seg_folder = QLineEdit("output/annotate/seg_imgs")
        self.flo.addRow("Output seg folder", self.output_seg_folder)

        self.vis_channels = QListWidget()
        self.vis_channels.insertItem(0, "0")
        self.vis_channels.insertItem(1, "1")
        self.vis_channels.insertItem(2, "2")
        self.vis_channels.insertItem(3, "3")
        self.vis_channels.setSelectionMode(2)
        self.vis_channels.setToolTip(
            "Choice of which channels user wants to view for seg"
        )
        self.vis_channels.item(0).setSelected(True)
        self.flo.addRow("Channels", self.vis_channels)

        # yaml save loc
        self.save_loc_input = QLineEdit("output/annotate/annotate.yaml")
        self.save_loc_input.setToolTip("Yaml save location")
        self.flo.addRow("yaml save location", self.save_loc_input)

        self.setLayout(self.flo)

        self.config = config

    def set_config(self, config):
        """Set the configuration file

        Args:
            config (dictionary) : Configuration dict"""

        config["x_bins"] = self.x_bins
        config["y_bins"] = self.y_bins
        config["z_bins"] = self.z_bins
        config["dim"] = self.dim
        config["plot"] = self.plot.isChecked()
        config["vis_interpolation"] = self.vis_interpolation.selectedItems()[0].text()
        config["input_folder"] = self.input_folder.text()
        config["histo_folder"] = self.histo_folder.text()
        config["output_folder"] = self.output_folder.text()
        config["drop_zero_label"] = self.drop_zero_label.isChecked()
        config["gt_label_map"] = {0: self.gt_label_map_zero, 1: self.gt_label_map_one}
        config["save_img"] = self.save_img.isChecked()
        config["save_threshold"] = self.save_threshold
        config["save_interpolate"] = self.save_interpolation.selectedItems()[0].text()
        config["background_one_colour"] = self.background_one_colour.isChecked()
        config["four_colour"] = self.four_colour.isChecked()
        config["alphas"] = [
            self.alpha_zero,
            self.alpha_one,
            self.alpha_two,
            self.alpha_three,
        ]
        config["alpha_seg"] = self.alpha_seg
        config["cmap_seg"] = [self.zero_cmap, self.one_cmap]
        config["fig_size"] = [self.fig_size_x, self.fig_size_y]
        config["output_seg_folder"] = self.output_seg_folder
        config["vis_channels"] = [
            item.text() for item in self.vis_channels.selectedItems()
        ]
        config["yaml_save_loc"] = self.save_loc_input.text()

        # check config is correct
        parse_config(config)

    def closeEvent(self, event):
        """When closing the widget this function will be overloaded.
            It will ask the user if they are sure, on closing the coords of the markers
                will be extracted
        Args:
            event (QEvent): Event triggered by click"""

        # Confirm user choice
        reply = QMessageBox()
        reply.setText("Are you sure you want to close? (Config is saved on closing)")
        reply.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        reply = reply.exec()
        if reply == QMessageBox.StandardButton.Yes:
            self.set_config(self.config)
            event.accept()
        else:
            event.ignore()


def config_gui():
    """Config gui
    This function opens up a GUI for user to specify
    the configuration
    List of files files can then be unchecked if users want to ignore

    Attributes:
        save_loc(string): Where to save the output
            .yaml file for the configuration
        files (list): List of files to be preprocessed"""

    app = QApplication([])  # sys.argv if need command line inputs
    # create widget
    config = {}
    widget = InputWidget(config)
    widget.show()
    app.exec()
    return config


def parse_config(config):
    """Parse config
    This function takes in the configuration .yaml
    file and checks all the necessary arguments are
    specified

    Attributes:
        config (yml file): The configuration
            .yaml file"""

    if sorted(config.keys()) != sorted(default_config_keys):
        raise ValueError(
            "Did not specify necessary default \
            configutation arguments"
        )
