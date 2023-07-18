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
    QPushButton,
    QFileDialog,
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
from PyQt5.QtCore import Qt
import yaml
import os
import json


# two options for default keys

default_config_keys_0 = [
    "vis_threshold",
    "vis_interpolate",
    "model",
    "diameter",
    "channels",
    "sum_chan",
    "test_files",  # new
    "channel",
    "alt_channel",
    "use_gpu",
    "img_threshold",
]

default_config_keys_1 = [
    "vis_threshold",
    "vis_interpolate",
    "user_model_path",
    "diameter",
    "channels",
    "sum_chan",
    "test_files",  # new
    "channel",
    "alt_channel",
    "use_gpu",
    "img_threshold",
]


class InputWidget(QWidget):
    """Input
    Widget to take in user configuration
    Child of QWidget.
    Args:
        None
    Attributes:

    """

    def __init__(self, config, proj_path):
        """Constructor
        Args:
            files (list) : List of files that will be preprocessed
            config (dict) : Dictionary containing the configuration
            proj_path (list) : List containing the path to the project folder
        """

        super().__init__()
        self.flo = QFormLayout()

        # Set project directory and parse the metadata
        h_box = QHBoxLayout()
        self.project_directory = QPushButton("Set project directory")
        self.project_directory.clicked.connect(self.load_project_directory)
        h_box.addWidget(self.project_directory)
        self.check_metadata = QPushButton("Check project metadata")
        self.check_metadata.clicked.connect(self.parse_metadata)
        h_box.addWidget(self.check_metadata)
        self.flo.addRow(h_box)

        # Load .yaml with button
        self.load_button = QPushButton("Load configuration")
        self.load_button.clicked.connect(self.load_yaml)
        self.flo.addRow(self.load_button)

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

        # Finished button
        self.finished_button = QPushButton("Finished!")
        self.finished_button.clicked.connect(self.close)
        self.flo.addRow(self.finished_button)

        self.setLayout(self.flo)

        self.config = config
        self.proj_path = proj_path

    def load_project_directory(self):
        """Load project directory from button"""

        # Load folder
        project_dir = QFileDialog.getExistingDirectory(
            self, "window", "/home/some/folder"
        )

        if project_dir == "":
            print("Empty project directory")

        self.proj_path.append(project_dir)

    def parse_metadata(self):
        """Check metadata for loaded in project directory"""

        # check project directory is populated
        if self.proj_path:
            # load in metadata
            with open(
                os.path.join(self.proj_path[0], "metadata.json"),
            ) as file:
                metadata = json.load(file)
                # metadata = json.dumps(metadata)
            # display metadata
            msg = QMessageBox()
            msg.setWindowTitle("Project metadata")
            meta_text = "".join(
                [f"{key} : {value} \n" for key, value in metadata.items()]
            )
            msg.setText(meta_text)
            msg.exec_()

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
                if sorted(load_config.keys()) == sorted(default_config_keys_0):
                    self.load_config(load_config)
                elif sorted(load_config.keys()) == sorted(default_config_keys_1):
                    self.load_config(load_config)
                else:
                    raise ValueError("Can't load in as keys don't match!")

    def load_config(self, load_config):
        """Load the config into the gui

        Args:
            load_config (yaml file): Config file
                to load into the gui"""

        self.vis_threshold.setText(str(load_config["vis_threshold"]))
        self.vis_interpolation.clearSelection()
        item = self.vis_interpolation.findItems(
            load_config["vis_interpolate"], Qt.MatchFlag.MatchExactly
        )
        item[0].setSelected(True)
        self.cellpose_model.setText(load_config["model"])
        self.cellpose_diameter.setText(str(load_config["diameter"]))
        self.first_channel.setText(str(load_config["channels"][0]))
        self.second_channel.setText(str(load_config["channels"][1]))
        self.sum_chan.setCheckState(load_config["sum_chan"])

    def set_config(self, config):
        """Set the configuration file

        Args:
            config (dictionary) : Configuration dict"""

        config["vis_threshold"] = float(self.vis_threshold.text())
        config["vis_interpolate"] = self.selectedItems()[0].text()
        config["model"] = self.cellpose_model.text()
        config["diameter"] = float(self.cellpose_diameter.text())
        config["channels"] = [
            int(self.first_channel.text()),
            int(self.second_channel.text()),
        ]
        config["sum_chan"] = self.sum_chan.isChecked()

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
    proj_path = []
    widget = InputWidget(config, proj_path)
    widget.show()
    app.exec()

    if not proj_path:
        raise ValueError("Project directory was not specified")

    return config, proj_path[0]


def parse_config(config):
    """Parse config
    This function takes in the configuration .yaml
    file and checks all the necessary arguments are
    specified

    Attributes:
        config (yml file): The configuration
            .yaml file"""

    if (sorted(config.keys()) != sorted(default_config_keys_0)) and (
        sorted(config.keys()) != sorted(default_config_keys_1)
    ):
        raise ValueError(
            "Did not specify necessary default \
            configutation arguments"
        )
