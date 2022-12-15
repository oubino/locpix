"""Membrane performance config

Module which defines functions for managing the config
files for membrane performance.
Including:

Creating a GUI for user to choose configuration
Parsing config file to check all specified"""

from PyQt5.QtWidgets import (
    QListView,
    QFrame,
    QAbstractItemView,
    QApplication,
    QWidget,
    QMessageBox,
    QLineEdit,
    QFormLayout,
    QListWidget,
    QPushButton,
    QFileDialog,
)
from PyQt5.QtGui import QDoubleValidator
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import yaml
import os

default_config_keys = [
    "maximise_choice",
    "vis_threshold",
    "vis_interpolate",
]


class QListDragAndDrop(QListWidget):
    def __init__(self):
        super(QListDragAndDrop, self).__init__()
        self.setFrameShape(QFrame.WinPanel)
        self.setFrameShadow(QFrame.Raised)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDrop)
        self.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.setSelectionMode(QAbstractItemView.MultiSelection)
        self.setMovement(QListView.Snap)
        self.setProperty("isWrapping", True)
        self.setWordWrap(True)
        self.setSortingEnabled(True)
        self.setAcceptDrops(True)


class InputWidget(QWidget):
    """Input
    Widget to take in user configuration
    Child of QWidget.
    Args:
        None
    Attributes:

    """

    def __init__(self, config, gt_file_path):
        """Constructor
        Args:
            gt_file_path (string) : Folder containing files that will be processed
            config (dict) : Dictionary containing the configuration
        """

        super().__init__()
        self.flo = QFormLayout()

        # Load .yaml with button
        self.load_button = QPushButton("Load yaml")
        self.load_button.clicked.connect(self.load_yaml)
        self.flo.addRow(self.load_button)

        # gt file path
        self.gt_file_path = gt_file_path

        # train and test files
        self.train_files = QListDragAndDrop()
        self.files = os.listdir(gt_file_path)
        self.files = [parquet.removesuffix(".parquet") for parquet in self.files]
        for index, value in enumerate(self.files):
            self.train_files.insertItem(index, value)
        self.test_files = QListDragAndDrop()
        self.flo.addRow("Train files", self.train_files)
        self.flo.addRow("Test files", self.test_files)

        self.maximise_choice = QListWidget()
        self.maximise_choice.insertItem(0, "recall")
        self.maximise_choice.insertItem(1, "f")
        self.maximise_choice.item(0).setSelected(True)
        self.maximise_choice.setToolTip("Which metric to maximise")
        self.flo.addRow("Maximise choice", self.maximise_choice)

        self.vis_threshold = QLineEdit("0")
        self.vis_threshold.setValidator(QDoubleValidator())
        self.flo.addRow("Vis threshold", self.vis_threshold)

        self.vis_interpolate = QListWidget()
        self.vis_interpolate.insertItem(0, "log2")
        self.vis_interpolate.insertItem(1, "log10")
        self.vis_interpolate.insertItem(2, "linear")
        self.vis_interpolate.item(0).setSelected(True)
        self.flo.addRow("vis interpolate", self.vis_interpolate)

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

        # check all files specified
        loaded_files = sorted(load_config["train_files"] + load_config["test_files"])
        if sorted(self.files) != loaded_files:
            print("Files need to match!")
        elif sorted(self.files) == loaded_files:
            self.train_files.clear()
            self.test_files.clear()
            for index, value in enumerate(load_config["train_files"]):
                self.train_files.insertItem(index, value)
            for index, value in enumerate(load_config["test_files"]):
                self.test_files.insertItem(index, value)

            self.maximise_choice.clearSelection()
            item = self.maximise_choice.findItems(
                load_config["maximise_choice"], Qt.MatchFlag.MatchExactly
            )
            item[0].setSelected(True)

            self.vis_threshold.setText(str(load_config["vis_threshold"]))
            self.vis_interpolate.clearSelection()
            item = self.vis_interpolate.findItems(
                load_config["vis_interpolate"], Qt.MatchFlag.MatchExactly
            )
            item[0].setSelected(True)

    def set_config(self, config):
        """Set the configuration file

        Args:
            config (dictionary) : Configuration dict"""

        config["train_files"] = [
            self.train_files.item(x).text() for x in range(self.train_files.count())
        ]
        config["test_files"] = [
            self.test_files.item(x).text() for x in range(self.test_files.count())
        ]
        config["maximise_choice"] = self.maximise_choice.selectedItems()[0].text()
        config["vis_threshold"] = float(self.vis_threshold.text())
        config["vis_interpolate"] = self.vis_interpolate.selectedItems()[0].text()

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


def config_gui(gt_file_path):
    """Config gui
    This function opens up a GUI for user to specify
    the configuration
    List of files files can then be unchecked if users want to ignore

    Attributes:
        gt_file_path (string): Folder containing files to be processed"""

    app = QApplication([])  # sys.argv if need command line inputs
    # create widget
    config = {}
    widget = InputWidget(config, gt_file_path)
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
