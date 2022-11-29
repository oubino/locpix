"""Cellpose config

Module which defines functions for managing the config
files for cellpose.
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
    "input_folder",
    "input_histo_folder",
    "markers_loc",
    "vis_threshold",
    "vis_interpolate",
    "model",
    "diameter",
    "channels",
    "sum_chan" "output_membrane_prob",
    "output_cell_df",
    "output_cell_img",
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

        self.input_folder = QLineEdit("output/preprocess/annotated")
        self.input_folder.setToolTip("Input folder")
        self.flo.addRow("Input folder", self.input_folder)

        self.input_histo_folder = QLineEdit("output/annotate/histos")
        self.input_histo_folder.setToolTip("Input folder for histograms")
        self.flo.addRow("Histo folder", self.input_histo_folder)

        self.markers_folder = QLineEdit("output/markers")
        self.markers_folder.setToolTip("Folder which has markers in it")
        self.flo.addRow("Markers folder", self.markers_folder)

        self.vis_threshold = QLineEdit("5")
        self.vis_threshold.setValidator(QDoubleValidator())
        self.flo.addRow("Vis threshold", self.vis_threshold)

        self.vis_interpolation = QListWidget()
        self.vis_interpolation.insertItem(0, "log2")
        self.vis_interpolation.insertItem(1, "log10")
        self.vis_interpolation.insertItem(2, "linear")
        self.vis_interpolation.item(0).setSelected(True)
        self.flo.addRow("vis interpolation", self.vis_interpolation)

        self.cellpose_model = QLineEdit("LC1")
        self.cellpose_model.setToolTip("Cellpose model")
        self.flo.addRow("Cellpose model", self.cellpose_model)

        self.cellpose_diameter = QLineEdit("50")
        self.cellpose_diameter.setValidator(QDoubleValidator())
        self.flo.addRow("Cellpose diameter", self.cellpose_diameter)

        h_box = QHBoxLayout()
        first_channel = QLabel("First channel: ")
        self.first_channel = QLineEdit("0")
        self.first_channel.setValidator(QIntValidator())
        h_box.addWidget(first_channel)
        h_box.addWidget(self.first_channel)
        second_channel = QLabel("Second channel: ")
        self.second_channel = QLineEdit("0")
        self.second_channel.setValidator(QIntValidator())
        h_box.addWidget(second_channel)
        h_box.addWidget(self.second_channel)
        help = (
            "Model is trained on two-channel images, where the first channel"
            "is the channel to segment and the second channel is an"
            "optional nuclear channel\n"
            "Options for each:\n"
            "a. 0=grayscale, 1=red, 2=green, 3=blue\n"
            "b. 0=None (will set to zero), 1=red, 2=green, 3=blue\n"
            "e.g. channels = [0,0] if you want to segment cells in grayscale"
        )
        self.first_channel.setToolTip(help)
        self.second_channel.setToolTip(help)
        self.flo.addRow("Gt label map", h_box)

        self.sum_chan = QCheckBox()
        self.sum_chan.setToolTip(
            "Whether to sum channels (currently only channel 0 and 1)"
        )
        self.flo.addRow("Sum channels", self.sum_chan)

        self.output_membrane_prob = QLineEdit("output/cellpose/membrane/prob_map")
        self.output_membrane_prob.setToolTip("Output membrane probability mask")
        self.flo.addRow("Output membrane probability mask", self.output_membrane_prob)

        self.output_cell_df = QLineEdit("output/cellpose/cell/seg_dataframes")
        self.output_cell_df.setToolTip("Output dataframe segmentation for cell")
        self.flo.addRow("Output dataframe segmentation for cell", self.output_cell_df)

        self.output_cell_img = QLineEdit("output/cellpose/cell/seg_img")
        self.output_cell_img.setToolTip("Output image of cell segmentation")
        self.flo.addRow("Output image of cell segmentation", self.output_cell_img)

        # yaml save loc
        self.save_loc_input = QLineEdit("output/cellpose/cellpose.yaml")
        self.save_loc_input.setToolTip("Yaml save location")
        self.flo.addRow("yaml save location", self.save_loc_input)

        self.setLayout(self.flo)

        self.config = config

    def set_config(self, config):
        """Set the configuration file

        Args:
            config (dictionary) : Configuration dict"""

        config["input_folder"] = self.input_folder.text()
        config["input_histo_folder"] = self.input_histo_folder.text()
        config["markers_loc"] = self.markers_folder.text()
        config["vis_threshold"] = self.vis_threshold
        config["vis_interpolate"] = self.selectedItems()[0].text()
        config["model"] = self.cellpose_model.text()
        config["diameter"] = self.cellpose_diameter
        config["channels"] = [self.first_channel, self.second_channel]
        config["sum_chan"] = self.sum_chan.isChecked()
        config["output_membrane_prob"] = self.output_membrane_prob.text()
        config["output_cell_df"] = self.output_cell_df.text()
        config["output_cell_img"] = self.output_cell_img.text()
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
