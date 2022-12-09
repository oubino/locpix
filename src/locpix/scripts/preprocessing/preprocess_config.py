"""Preprocess config

Module which defines functiosn for managing the config
files for preprocessing.
Including:

Creating a GUI for user to choose configuration
Parsing config file to check all specified"""

from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QMessageBox,
    QLineEdit,
    QFormLayout,
    QCheckBox,
    QListWidget,
    QPushButton,
    QFileDialog,
)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt

import yaml

default_config_keys = [
    "x_col",
    "y_col",
    "z_col",
    "channel_col",
    "frame_col",
    "dim",
    # "output_folder",
    "channel_choice",
    "drop_pixel_col",
    "include_files",
    # "yaml_save_loc",
]


class InputWidget(QWidget):
    """Input
    Widget to take in user configuration
    Child of QWidget.
    Args:
        None
    Attributes:

    """

    def __init__(self, files, config):
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

        # The following are the names of the
        # x column, y column, z column if present, channel, frame,
        # in the csvs being processed
        self.x_col = QLineEdit("X (nm)")  # x col
        self.x_col.setToolTip("Name of the x column")
        self.flo.addRow("X column", self.x_col)
        self.y_col = QLineEdit("Y (nm)")  # y col
        self.y_col.setToolTip("Name of the y column")
        self.flo.addRow("Y column", self.y_col)
        self.z_col = QLineEdit(None)  # z col
        self.z_col.setToolTip("Name of the z column (leave empty if not present)")
        self.flo.addRow("Z column", self.z_col)
        self.chan_col = QLineEdit("Channel")
        self.chan_col.setToolTip("Name of the channel column")
        self.flo.addRow("Channel column", self.chan_col)
        self.frame_col = QLineEdit("Frame")
        self.frame_col.setToolTip("Name of the frame column")
        self.flo.addRow("Frame column", self.frame_col)

        # The number of dimensions to consider
        # If 2 only deals with x and y
        # If 3 will read in and deal with z as well (currently not fully supported)
        self.dim = QLineEdit("2")
        self.dim.setToolTip(
            "The number of dimensions to consider If 2 only deals with x and y\
            If 3 will read in and deal wih\
            z as well (currently not fully supported)"
        )
        self.dim.setValidator(QIntValidator())
        self.flo.addRow("Dimensions", self.dim)

        # output folder
        # self.output_folder = QLineEdit("output/preprocessed/no_gt_label")
        # self.output_folder.setToolTip("Name of output folder")
        # self.flo.addRow("Output folder", self.output_folder)

        # choice of which channels user wants to consider
        # if null considers all
        self.channel_choice = QListWidget()
        self.channel_choice.insertItem(0, "0")
        self.channel_choice.insertItem(1, "1")
        self.channel_choice.insertItem(2, "2")
        self.channel_choice.insertItem(3, "3")
        self.channel_choice.setSelectionMode(2)
        self.channel_choice.setToolTip(
            "Choice of which channels user wants to consider"
        )
        self.flo.addRow("Channels", self.channel_choice)

        # whether to not drop the column containing
        # pixel
        self.drop_pixel_col = QCheckBox()
        self.drop_pixel_col.setToolTip(
            "whether to not drop the column containing pixel"
        )
        self.flo.addRow("Drop pixel col", self.drop_pixel_col)

        # files to include
        self.include_files = QListWidget()
        for index, value in enumerate(files):
            self.include_files.insertItem(index, value)
            self.include_files.setSelectionMode(2)
        self.include_files.setToolTip("Files to include")
        self.flo.addRow("Files to include", self.include_files)

        # yaml save loc
        # self.save_loc_input = QLineEdit("output/preprocess/preprocess.yaml")
        # self.save_loc_input.setToolTip("Yaml save location")
        # self.flo.addRow("yaml save location", self.save_loc_input)

        self.setLayout(self.flo)
        # self.include_files.selectAll()

        self.files = files
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

        self.x_col.setText(load_config["x_col"])
        self.y_col.setText(load_config["y_col"])
        self.z_col.setText(load_config["z_col"])
        self.chan_col.setText(load_config["channel_col"])
        self.frame_col.setText(load_config["frame_col"])
        self.dim.setText(str(load_config["dim"]))
        # self.output_folder.setText(load_config["output_folder"])

        self.channel_choice.clearSelection()
        for chan in load_config["channel_choice"]:
            item = self.channel_choice.findItems(str(chan), Qt.MatchFlag.MatchExactly)
            item[0].setSelected(True)

        self.drop_pixel_col.setCheckState(load_config["drop_pixel_col"])

        self.include_files.clearSelection()
        for file in load_config["include_files"]:
            item = self.include_files.findItems(file, Qt.MatchFlag.MatchExactly)
            if item:
                item[0].setSelected(True)

        # self.save_loc_input.setText(load_config["yaml_save_loc"])

    def set_config(self, config):
        """Set the configuration file

        Args:
            config (dictionary) : Configuration dict"""

        config["x_col"] = self.x_col.text()
        config["y_col"] = self.y_col.text()
        config["z_col"] = self.z_col.text()
        config["channel_col"] = self.chan_col.text()
        config["frame_col"] = self.frame_col.text()
        config["dim"] = int(self.dim.text())
        # config["output_folder"] = self.output_folder.text()
        chan_list = self.channel_choice.selectedItems()
        chan_list = [int(item.text()) for item in chan_list]
        config["channel_choice"] = chan_list
        config["drop_pixel_col"] = self.drop_pixel_col.isChecked()
        include_files = self.include_files.selectedItems()
        config["include_files"] = [item.text() for item in include_files]
        # config["yaml_save_loc"] = self.save_loc_input.text()

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


def config_gui(files):
    """Config gui
    This function opens up a GUI for user to specify
    the configuration
    List of files files can then be unchecked if users want to ignore

    Attributes:
        files (list): List of files to be preprocessed"""

    app = QApplication([])  # sys.argv if need command line inputs
    # create widget
    config = {}
    widget = InputWidget(files, config)
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
