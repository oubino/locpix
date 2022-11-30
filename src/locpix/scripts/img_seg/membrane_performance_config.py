"""Membrane performance config

Module which defines functions for managing the config
files for membrane performance.
Including:

Creating a GUI for user to choose configuration
Parsing config file to check all specified"""

from PyQt5.QtWidgets import (
    QLabel,
    QListView,
    QFrame,
    QAbstractItemView,
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
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import yaml
import os

default_config_keys = [
    "train_files",
    "test_files",
    "maximise_choice",
    "input_histo_folder",
    "gt_file_path",
    "classic_seg_folder",
    "cellpose_seg_folder",
    "ilastik_seg_folder",
    "output_classic_df_folder",
    "output_cellpose_df_folder",
    "output_ilastik_df_folder",
    "output_classic_seg_imgs",
    "output_cellpose_seg_imgs",
    "output_ilastik_seg_imgs",
    "output_train_classic_pr",
    "output_train_cellpose_pr",
    "output_train_ilastik_pr",
    "output_test_classic_pr",
    "output_test_cellpose_pr",
    "output_test_ilastik_pr",
    "output_metrics_classic",
    "output_metrics_cellpose",
    "output_metrics_ilastik",
    "output_conf_matrix_classic",
    "output_conf_matrix_cellpose",
    "output_conf_matrix_ilastik",
    "output_overlay_pr_curves",
    "vis_threshold",
    "vis_interpolate",
    "yaml_save_loc",
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

        self.input_histo_folder = QLineEdit("output/annotate/histos")
        self.input_histo_folder.setToolTip("Input histogram folder")
        self.flo.addRow("Input histogram folder", self.input_histo_folder)

        self.classic_seg_folder = QLineEdit("output/classic/membrane/prob_map")
        self.classic_seg_folder.setToolTip("Location of classic segmentation")
        self.flo.addRow("Classic segmentation", self.classic_seg_folder)

        self.cellpose_seg_folder = QLineEdit("output/cellpose/membrane/prob_map")
        self.cellpose_seg_folder.setToolTip("Location of cellpose segmentation")
        self.flo.addRow("cellpose segmentation", self.cellpose_seg_folder)

        self.ilastik_seg_folder = QLineEdit("output/ilastik/output/membrane/prob_map")
        self.ilastik_seg_folder.setToolTip("Location of ilastik segmentation")
        self.flo.addRow("ilastik segmentation", self.ilastik_seg_folder)

        self.classic_df_folder = QLineEdit(
            "output/membrane_performance/classic/membrane/seg_dataframes"
        )
        self.classic_df_folder.setToolTip("Where to save output parquet files")
        self.flo.addRow("Classic segmented dataframe", self.classic_df_folder)

        self.cellpose_df_folder = QLineEdit(
            "output/membrane_performance/cellpose/membrane/seg_dataframes"
        )
        self.cellpose_df_folder.setToolTip("Where to save output parquet files")
        self.flo.addRow("cellpose segmented dataframe", self.cellpose_df_folder)

        self.ilastik_df_folder = QLineEdit(
            "output/membrane_performance/ilastik/membrane/seg_dataframes"
        )
        self.ilastik_df_folder.setToolTip("Where to save output parquet files")
        self.flo.addRow("ilastik segmented dataframe", self.ilastik_df_folder)

        self.output_classic_seg_imgs = QLineEdit(
            "output/membrane_performance/classic/membrane/seg_images"
        )
        self.output_classic_seg_imgs.setToolTip(
            "Where to save output images of segmentation"
        )
        self.flo.addRow("Output classic segmented images", self.output_classic_seg_imgs)

        self.output_cellpose_seg_imgs = QLineEdit(
            "output/membrane_performance/cellpose/membrane/seg_images"
        )
        self.output_cellpose_seg_imgs.setToolTip(
            "Where to save output images of segmentation"
        )
        self.flo.addRow(
            "Output cellpose segmented images", self.output_cellpose_seg_imgs
        )

        self.output_ilastik_seg_imgs = QLineEdit(
            "output/membrane_performance/ilastik/membrane/seg_images"
        )
        self.output_ilastik_seg_imgs.setToolTip(
            "Where to save output images of segmentation"
        )
        self.flo.addRow("Output ilastik segmented images", self.output_ilastik_seg_imgs)

        self.output_train_classic_pr = QLineEdit(
            "output/membrane_performance/classic/membrane/train_pr"
        )
        self.output_train_classic_pr.setToolTip("Where to save output train pr curves")
        self.flo.addRow("Output classic pr curves", self.output_train_classic_pr)

        self.output_train_cellpose_pr = QLineEdit(
            "output/membrane_performance/cellpose/membrane/train_pr"
        )
        self.output_train_cellpose_pr.setToolTip("Where to save output train pr curves")
        self.flo.addRow("Output cellpose pr curves", self.output_train_cellpose_pr)

        self.output_train_ilastik_pr = QLineEdit(
            "output/membrane_performance/ilastik/membrane/train_pr"
        )
        self.output_train_ilastik_pr.setToolTip("Where to save output train pr curves")
        self.flo.addRow("Output ilastik pr curves", self.output_train_ilastik_pr)

        self.output_test_classic_pr = QLineEdit(
            "output/membrane_performance/classic/membrane/test_pr"
        )
        self.output_test_classic_pr.setToolTip("Where to save output test pr curves")
        self.flo.addRow("Output classic pr curves", self.output_test_classic_pr)

        self.output_test_cellpose_pr = QLineEdit(
            "output/membrane_performance/cellpose/membrane/test_pr"
        )
        self.output_test_cellpose_pr.setToolTip("Where to save output test pr curves")
        self.flo.addRow("Output cellpose pr curves", self.output_test_cellpose_pr)

        self.output_test_ilastik_pr = QLineEdit(
            "output/membrane_performance/ilastik/membrane/test_pr"
        )
        self.output_test_ilastik_pr.setToolTip("Where to save output test pr curves")
        self.flo.addRow("Output ilastik pr curves", self.output_test_ilastik_pr)

        self.output_metrics_classic = QLineEdit(
            "output/membrane_performance/classic/membrane/metrics"
        )
        self.output_metrics_classic.setToolTip(
            "Where to save output performance metrics"
        )
        self.flo.addRow(
            "Output classic performance metrics", self.output_metrics_classic
        )

        self.output_metrics_cellpose = QLineEdit(
            "output/membrane_performance/cellpose/membrane/metrics"
        )
        self.output_metrics_cellpose.setToolTip(
            "Where to save output performance metrics"
        )
        self.flo.addRow(
            "Output cellpose performance metrics", self.output_metrics_cellpose
        )

        self.output_metrics_ilastik = QLineEdit(
            "output/membrane_performance/ilastik/membrane/metrics"
        )
        self.output_metrics_ilastik.setToolTip(
            "Where to save output performance metrics"
        )
        self.flo.addRow(
            "Output ilastik performance metrics", self.output_metrics_ilastik
        )

        self.output_conf_matrix_classic = QLineEdit(
            "output/membrane_performance/classic/membrane/conf_matrix"
        )
        self.output_conf_matrix_classic.setToolTip(
            "where to save output confusion matrix"
        )
        self.flo.addRow(
            "Output classic confusion matrix", self.output_conf_matrix_classic
        )

        self.output_conf_matrix_cellpose = QLineEdit(
            "output/membrane_performance/cellpose/membrane/conf_matrix"
        )
        self.output_conf_matrix_cellpose.setToolTip(
            "where to save output confusion matrix"
        )
        self.flo.addRow(
            "Output cellpose confusion matrix", self.output_conf_matrix_cellpose
        )

        self.output_conf_matrix_ilastik = QLineEdit(
            "output/membrane_performance/ilastik/membrane/conf_matrix"
        )
        self.output_conf_matrix_ilastik.setToolTip(
            "where to save output confusion matrix"
        )
        self.flo.addRow(
            "Output ilastik confusion matrix", self.output_conf_matrix_ilastik
        )

        self.output_overlay_pr_curves = QLineEdit(
            "output/membrane_performance/overlaid_pr_curves"
        )
        self.output_overlay_pr_curves.setToolTip("Where to save overlaid pr curves")
        self.flo.addRow("Output overlaid pr curves", self.output_overlay_pr_curves)

        self.vis_threshold = QLineEdit("0")
        self.vis_threshold.setValidator(QDoubleValidator())
        self.flo.addRow("Vis threshold", self.vis_threshold)

        self.vis_interpolate = QListWidget()
        self.vis_interpolate.insertItem(0, "log2")
        self.vis_interpolate.insertItem(1, "log10")
        self.vis_interpolate.insertItem(2, "linear")
        self.vis_interpolate.item(0).setSelected(True)
        self.flo.addRow("vis interpolate", self.vis_interpolate)

        # yaml save loc
        self.yaml_save_loc = QLineEdit(
            "output/membrane_performance/membrane_performance.yaml"
        )
        self.yaml_save_loc.setToolTip("Yaml save location")
        self.flo.addRow("yaml save location", self.yaml_save_loc)

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

        # check all files specified 
        loaded_files = sorted(load_config["train_files"] + load_config["test_files"])
        if sorted(self.files) != loaded_files:
            print('Files need to match!')
        elif sorted(self.files) == loaded_files:
            self.train_files.clear()
            self.test_files.clear()
            for index, value in enumerate(load_config["train_files"]):
                self.train_files.insertItem(index, value)
            for index, value in enumerate(load_config["test_files"]):
                self.test_files.insertItem(index, value)

            self.maximise_choice.clearSelection()
            item = self.maximise_choice.findItems(load_config["maximise_choice"], Qt.MatchFlag.MatchExactly)
            item[0].setSelected(True)

            self.input_histo_folder.setText(load_config["input_histo_folder"])
            self.gt_file_path= load_config["gt_file_path"]

            self.classic_seg_folder.setText(load_config["classic_seg_folder"])
            self.cellpose_seg_folder.setText(load_config["cellpose_seg_folder"])
            self.ilastik_seg_folder.setText(load_config["ilastik_seg_folder"])

            self.classic_df_folder.setText(load_config["output_classic_df_folder"])
            self.cellpose_df_folder.setText(load_config["output_cellpose_df_folder"])
            self.ilastik_df_folder.setText(load_config["output_ilastik_df_folder"])

            self.output_classic_seg_imgs.setText(load_config["output_classic_seg_imgs"])
            self.output_cellpose_seg_imgs.setText(load_config["output_cellpose_seg_imgs"])
            self.output_ilastik_seg_imgs.setText(load_config["output_ilastik_seg_imgs"])

            self.output_train_classic_pr.setText(load_config["output_train_classic_pr"])
            self.output_train_cellpose_pr.setText(load_config["output_train_cellpose_pr"])
            self.output_train_ilastik_pr.setText(load_config["output_train_ilastik_pr"])

            self.output_test_classic_pr.setText(load_config["output_test_classic_pr"])
            self.output_test_cellpose_pr.setText(load_config["output_test_cellpose_pr"])
            self.output_test_ilastik_pr.setText(load_config["output_test_ilastik_pr"])

            self.output_metrics_classic.setText(load_config["output_metrics_classic"])
            self.output_metrics_cellpose.setText(load_config["output_metrics_cellpose"])
            self.output_metrics_ilastik.setText(load_config["output_metrics_ilastik"])

            self.output_conf_matrix_classic.setText( load_config["output_conf_matrix_classic"])
            self.output_conf_matrix_cellpose.setText(load_config["output_conf_matrix_cellpose"])
            self.output_conf_matrix_ilastik.setText(load_config["output_conf_matrix_ilastik"])

            self.output_overlay_pr_curves.setText(load_config["output_overlay_pr_curves"])
                
            self.vis_threshold.setText(str(load_config["vis_threshold"]))
            self.vis_interpolate.clearSelection()
            item = self.vis_interpolate.findItems(load_config["vis_interpolate"], Qt.MatchFlag.MatchExactly)
            item[0].setSelected(True)
            self.yaml_save_loc.setText(load_config["yaml_save_loc"])


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
        config["input_histo_folder"] = self.input_histo_folder.text()
        config["gt_file_path"] = self.gt_file_path

        config["classic_seg_folder"] = self.classic_seg_folder.text()
        config["cellpose_seg_folder"] = self.cellpose_seg_folder.text()
        config["ilastik_seg_folder"] = self.ilastik_seg_folder.text()

        config["output_classic_df_folder"] = self.classic_df_folder.text()
        config["output_cellpose_df_folder"] = self.cellpose_df_folder.text()
        config["output_ilastik_df_folder"] = self.ilastik_df_folder.text()

        config["output_classic_seg_imgs"] = self.output_classic_seg_imgs.text()
        config["output_cellpose_seg_imgs"] = self.output_cellpose_seg_imgs.text()
        config["output_ilastik_seg_imgs"] = self.output_ilastik_seg_imgs.text()

        config["output_train_classic_pr"] = self.output_train_classic_pr.text()
        config["output_train_cellpose_pr"] = self.output_train_cellpose_pr.text()
        config["output_train_ilastik_pr"] = self.output_train_ilastik_pr.text()

        config["output_test_classic_pr"] = self.output_test_classic_pr.text()
        config["output_test_cellpose_pr"] = self.output_test_cellpose_pr.text()
        config["output_test_ilastik_pr"] = self.output_test_ilastik_pr.text()

        config["output_metrics_classic"] = self.output_metrics_classic.text()
        config["output_metrics_cellpose"] = self.output_metrics_cellpose.text()
        config["output_metrics_ilastik"] = self.output_metrics_ilastik.text()

        config["output_conf_matrix_classic"] = self.output_conf_matrix_classic.text()
        config["output_conf_matrix_cellpose"] = self.output_conf_matrix_cellpose.text()
        config["output_conf_matrix_ilastik"] = self.output_conf_matrix_ilastik.text()

        config["output_overlay_pr_curves"] = self.output_overlay_pr_curves.text()
        config["vis_threshold"] = int(self.vis_threshold.text())
        config["vis_interpolate"] = self.vis_interpolate.selectedItems()[0].text()

        config["yaml_save_loc"] = self.yaml_save_loc.text()

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
