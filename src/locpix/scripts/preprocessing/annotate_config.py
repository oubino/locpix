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
    QPushButton,
    QFileDialog,
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt
import yaml

default_config_keys = [
    "x_bins",
    "y_bins",
    "z_bins",
    "dim",
    "plot",
    "vis_interpolation",
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
    "vis_channels",
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

        # Load .yaml with button
        self.load_button = QPushButton("Load yaml")
        self.load_button.clicked.connect(self.load_yaml)
        self.flo.addRow(self.load_button)

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
            "Interpolation applied to the histogram when visualising"
            "the image of the histogram"
        )
        self.flo.addRow("Vis interpolation", self.vis_interpolation)

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

        self.setLayout(self.flo)

        self.config = config

    def load_yaml(self):
        """Load the yaml"""

        # Load yaml
        fname = QFileDialog.getOpenFileName(
            self, "Open file", "/home/some/folder", "Yaml (*.yaml)"
        )

        fname = str(fname[0])
        if fname != "":
            with open(fname, "r") as ymlfile:
                load_config = yaml.safe_load(ymlfile)
                if sorted(load_config.keys()) == sorted(default_config_keys):
                    self.load_config(load_config)
                else:
                    print("Can't load in as keys don't match!")

    def load_config(self, load_config):
        """Load the config into the gui

        Args:
            load_config (yaml file): Config file
                to load into the gui"""

        self.x_bins.setText(str(load_config["x_bins"]))
        self.y_bins.setText(str(load_config["y_bins"]))
        self.z_bins.setText(str(load_config["z_bins"]))
        self.dim.setText(str(load_config["dim"]))
        self.plot.setCheckState(load_config["plot"])
        self.vis_interpolation.clearSelection()
        item = self.vis_interpolation.findItems(
            load_config["vis_interpolation"], Qt.MatchFlag.MatchExactly
        )
        item[0].setSelected(True)
        self.drop_zero_label.setCheckState(load_config["drop_zero_label"])
        self.gt_label_map_zero.setText(str(load_config["gt_label_map"][0]))
        self.gt_label_map_one.setText(str(load_config["gt_label_map"][1]))
        self.save_img.setCheckState(load_config["save_img"])
        self.save_threshold.setText(str(load_config["save_threshold"]))
        self.save_interpolation.clearSelection()
        item = self.save_interpolation.findItems(
            load_config["save_interpolate"], Qt.MatchFlag.MatchExactly
        )
        item[0].setSelected(True)
        self.background_one_colour.setCheckState(load_config["background_one_colour"])
        self.four_colour.setCheckState(load_config["four_colour"])
        alphas = load_config["alphas"]
        self.alpha_zero.setText(str(alphas[0]))
        self.alpha_one.setText(str(alphas[1]))
        self.alpha_two.setText(str(alphas[2]))
        self.alpha_three.setText(str(alphas[3]))
        self.alpha_seg.setText(str(load_config["alpha_seg"]))
        self.zero_cmap.setText(load_config["cmap_seg"][0])
        self.one_cmap.setText(load_config["cmap_seg"][1])
        self.fig_size_x.setText(str(load_config["fig_size"][0]))
        self.fig_size_y.setText(str(load_config["fig_size"][1]))
        self.vis_channels.clearSelection()
        for chan in load_config["vis_channels"]:
            item = self.vis_channels.findItems(str(chan), Qt.MatchFlag.MatchExactly)
            item[0].setSelected(True)

    def set_config(self, config):
        """Set the configuration file

        Args:
            config (dictionary) : Configuration dict"""

        config["x_bins"] = int(self.x_bins.text())
        config["y_bins"] = int(self.y_bins.text())
        config["z_bins"] = int(self.z_bins.text())
        config["dim"] = int(self.dim.text())
        config["plot"] = self.plot.isChecked()
        config["vis_interpolation"] = self.vis_interpolation.selectedItems()[0].text()
        config["drop_zero_label"] = self.drop_zero_label.isChecked()
        config["gt_label_map"] = {
            0: self.gt_label_map_zero.text(),
            1: self.gt_label_map_one.text(),
        }
        config["save_img"] = self.save_img.isChecked()
        config["save_threshold"] = float(self.save_threshold.text())
        config["save_interpolate"] = self.save_interpolation.selectedItems()[0].text()
        config["background_one_colour"] = self.background_one_colour.isChecked()
        config["four_colour"] = self.four_colour.isChecked()
        config["alphas"] = [
            float(self.alpha_zero.text()),
            float(self.alpha_one.text()),
            float(self.alpha_two.text()),
            float(self.alpha_three.text()),
        ]
        config["alpha_seg"] = float(self.alpha_seg.text())
        config["cmap_seg"] = [self.zero_cmap.text(), self.one_cmap.text()]
        config["fig_size"] = [int(self.fig_size_x.text()), int(self.fig_size_y.text())]
        config["vis_channels"] = [
            int(item.text()) for item in self.vis_channels.selectedItems()
        ]

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
        None"""

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
