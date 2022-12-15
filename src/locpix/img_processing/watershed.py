"""Module which allows for the watershed algorithm to
be implemented"""

# imports for widget
from PyQt5.QtWidgets import QGridLayout, QWidget, QGraphicsScene, QGraphicsView
from PyQt5.QtWidgets import QGraphicsPixmapItem, QMessageBox, QPushButton
from PyQt5.QtGui import QPixmap, QPen, QBrush
from PyQt5.QtGui import QImage, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication

from skimage.segmentation import watershed
import numpy as np
import warnings


class WatershedWidget(QWidget):
    """WatershedWidget

    Widget to visualise an image as Grayscale 8 bit, then annotate with the seeds
    used for watershed.
    Child of QWidget.
    Left click places a seed, right click removes the seed.
    On closing the seed coordinates are saved

    Args:
        None

    Attributes:
        scene(QGraphicsScene): This holds all the 2D items (img, all ellipses added)
        view(QGraphicsView): Visualise the scene
        pixmap_item(QGraphicsPixmapItem): Image representation that can be used as
             a paint device
        help_button (QPushButton): Displays help for user
        pen(QPen): How the QPainter draws lines and outlines of shapes
        brush(QBrush): How to fill the shapes drawn by the painter
        marker_coords(list): List containing coordinates of the seeds for watershed

    """

    def __init__(self, img, coords=[], file_name="Image"):
        """Constructor

        Args:
             img (int8 numpy array): Numpy array range [0 255] which will be
                 converted to correct format for pixmap
             coords (list): Seed coordinate list which will pass by reference
             file_name (img name): Will display the image name
        """

        super().__init__()

        # create scene to add to and visualise it
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.setWindowTitle(file_name)

        # create layout manager add the widget to it
        self.layout = QGridLayout()
        self.layout.addWidget(self.view, 0, 0)
        self.setLayout(self.layout)  # set layout manager for widget

        # image in format pyqt needs
        # create pixmap item and add to scene
        img = img.copy()
        image = QImage(img, img.shape[1], img.shape[0], QImage.Format.Format_Grayscale8)
        self.pixmap_item = QGraphicsPixmapItem(QPixmap(image))
        self.scene.addItem(self.pixmap_item)

        # instructions button
        self.help_button = QPushButton("Help")
        self.help_button.setCheckable(True)
        self.help_button.toggle()
        self.help_button.clicked.connect(self.help_button_state)
        self.layout.addWidget(self.help_button, 0, 1)

        # overload mouse press event for pixmap item to our function
        self.pixmap_item.mousePressEvent = self.label_cell

        # set pen and brush for labels
        self.pen = QPen(QColor("red"))
        self.brush = QBrush(QColor("red"))

        # create empty set of marker coordinates
        self.marker_coords = coords

    def help_button_state(self):
        """If help button is clicked display alert widget which details the help"""

        alert = QMessageBox()
        alert.setWindowTitle("Help")
        alert.setText(
            "Left click: add a seed \n"
            "Right click: remove a seed\n"
            "When closing all seed coordinates will be saved!"
        )
        alert.exec()

    def label_cell(self, event):
        """On mouse click this definition will be called

        Left click: add seed
        Right click: remove seed

        Args:
            event (QEvent): Event triggered by click"""

        # add label on left click
        if event.button() == Qt.MouseButton.LeftButton:
            self.scene.addEllipse(
                event.pos().x(), event.pos().y(), 10, 10, self.pen, self.brush
            )
        # delete label on right click
        if event.button() == Qt.MouseButton.RightButton:
            # get list of items at that location
            items = self.scene.items(
                event.pos().x(),
                event.pos().y(),
                1,
                1,
                Qt.ItemSelectionMode.IntersectsItemShape,
                Qt.SortOrder.AscendingOrder,
            )
            # if item is ellipse (type 4) remove it (don't remove image!)
            for item in items:
                if item.type() == 4:
                    self.scene.removeItem(item)

    def get_coords(self):
        """On closing the widget the coordinates of the remaining seeds are extracted"""

        # perform when closed
        items = self.scene.items()
        warnings.warn(
            "Not accurate - needs to be adjusted before exact coordinate is returned"
        )

        # if ellipse add item coordinates to list
        for item in items:
            # if ellipse
            if item.type() == 4:
                # note not sure if lines up correct position - not issue here
                # i.e. if click at 100,240 is marker placed at 100,240?
                # and if its the centre of the marker? shouldn't we return
                # centre of marker
                # as the location
                # flag as issue!!!
                # but would be for accurate segmentation
                pos = item.pos()
                pos += item.rect().topLeft()
                # marker_coords is [H,W] relative to top left of image
                # i.e. if click bottom left of image marker_coords = [470,10]
                # this is what is returned to user
                self.marker_coords.append((int(pos.y()), int(pos.x())))

    def closeEvent(self, event):
        """When closing the widget this function will be overloaded.

        It will ask the user if they are sure, on closing the coords of the markers
        will be extracted

        Args:
            event (QEvent): Event triggered by click"""

        # Confirm user choice
        reply = QMessageBox()
        reply.setText("Are you sure you want to close? (Markers are saved on closing)")
        reply.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        reply = reply.exec()
        if reply == QMessageBox.StandardButton.Yes:
            # get the coordinates of the markers
            self.get_coords()
            event.accept()
        else:
            event.ignore()


def get_markers(img, file_name="Image") -> list:
    """Get markers for the image for watershed_segment using PyQt6 widget
    Image will be converted to [0 255] greyscale image for compatibility
    with PyQt6 widget

    Args:
        img (np.ndarray): Image for which need markers for watershed
        file_name (string): Name of the file

    Returns:
        marker_coords (list): List of tuples containing coordinates of markers (h,w)
            relative to top left of image"""

    app = QApplication([])  # sys.argv if need command line inputs
    # scale img to [0 255] and convert to int8
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    img = img.astype("uint8")
    # marker coords
    marker_coords = []
    # create widget
    widget = WatershedWidget(img, coords=marker_coords, file_name=file_name)
    widget.show()
    app.exec()
    # note marker coords are (h,w) relative to top left of image i.e.
    # if click bottom left of image
    # marker_coords = [470,10]
    return marker_coords


def watershed_segment(img, file_name="Image", coords=None) -> np.ndarray:
    """Perform watershed segmentation on image - with option either to provide
    coordinates of markers (coords)
    or obtain annotation using a widget.

    Args:
        img (np.ndarray): Image which performing watershed on
        file_name (string): Name of the image
        coords (list): List of tuples where each tuple represents coordinate
            of a marker (x,y)

    Returns:
        labels (np.ndarray): Numpy array containing integer labels, each
        representing different segmented region of the input
    """

    markers = np.zeros(img.shape, dtype="int32")
    # get markers if not provided
    if coords is None:
        coords = get_markers(img, file_name)
    # coords are (h,w) in image space
    # markers[row,col] where row is h-3:h+3 and col is w-3:w+3
    # i.e. imagine if select marker in bottom left of image
    # in image space [h,w] with origin at top this would be e.g. (470,10)
    # return coordinate (470,10) from get_coords
    # markers [467:473,7:13] is populated i.e. height of 470 ish
    # and width 10 ish is populated
    # this ensures img and marker coords are in same space
    for index, coord in enumerate(coords):
        markers[coord[0] - 3 : coord[0] + 3, coord[1] - 3 : coord[1] + 3] = int(
            index + 1
        )
    # perform watershed
    labels = watershed(img, markers=markers)
    return labels
