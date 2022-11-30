"""Get markers config

Module which defines functions for managing the config
files for get markers.
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
    "input_folder",
    "input_histo_folder",
    "markers_folder",
    "vis_threshold",
    "vis_interpolate",
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

        # Load .yaml with button
        self.load_button = QPushButton("Load yaml")
        self.load_button.clicked.connect(self.load_yaml)
        self.flo.addRow(self.load_button)

        self.input_folder = QLineEdit("output/preprocess/no_gt_label")
        self.input_folder.setToolTip("Input folder")
        self.flo.addRow("Input folder", self.input_folder)

        self.input_histo_folder = QLineEdit("output/annotate/histos")
        self.input_histo_folder.setToolTip("Input folder for histograms")
        self.flo.addRow("Histo folder", self.input_histo_folder)

        self.markers_folder = QLineEdit("output/markers")
        self.markers_folder.setToolTip("Folder to save markers to")
        self.flo.addRow("Markers folder", self.markers_folder)

        self.vis_threshold = QLineEdit("0")
        self.vis_threshold.setValidator(QDoubleValidator())
        self.flo.addRow("Vis threshold", self.vis_threshold)

        self.vis_interpolation = QListWidget()
        self.vis_interpolation.insertItem(0, "log2")
        self.vis_interpolation.insertItem(1, "log10")
        self.vis_interpolation.insertItem(2, "linear")
        self.vis_interpolation.item(0).setSelected(True)
        self.flo.addRow("vis interpolation", self.vis_interpolation)

        # yaml save loc
        self.save_loc_input = QLineEdit("output/markers/get_markers.yaml")
        self.save_loc_input.setToolTip("Yaml save location")
        self.flo.addRow("yaml save location", self.save_loc_input)

        self.setLayout(self.flo)

        self.config = config

    def load_yaml(self):
        """Load the yaml"""

        # Load yaml
        fname = QFileDialog.getOpenFileName(self, 'Open file', 
                "/home/some/folder","Yaml (*.yaml)")

        fname = str(fname[0])
        if fname != '':
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

        self.input_folder.setText(load_config["input_folder"])
        self.input_histo_folder.setText(load_config["input_histo_folder"])
        self.markers_folder.setText(load_config["markers_folder"])
        self.vis_threshold = load_config["vis_threshold"]
        self.vis_interpolation.clearSelection()
        item = self.vis_interpolation.findItems(load_config["vis_interpolate"], Qt.MatchFlag.MatchExactly)
        item[0].setSelected(True)
        self.save_loc_input.setText(load_config["yaml_save_loc"])

    def set_config(self, config):
        """Set the configuration file

        Args:
            config (dictionary) : Configuration dict"""

        config["input_folder"] = self.input_folder.text()
        config["input_histo_folder"] = self.input_histo_folder.text()
        config["markers_folder"] = self.markers_folder.text()
        config["vis_threshold"] = self.vis_threshold
        config["vis_interpolate"] = self.vis_interpolation.selectedItems()[0].text()
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
