######
locpix
######

**locpix** is a Python library for analysing point cloud data from SMLM.
This includes the following functionality:

#. :ref:`preprocess` : Initialises project and converts .csv files representing SMLM data (point cloud) into .parquet files, **necessary for this software**
#. :ref:`annotate` : Generating histograms from the SMLM data and manually annotating these histograms to extract relevant localisations
#. :ref:`get-markers` : Labelling histogram with seeds for watershed algorithm
#. Segmentation:

   #. :ref:`classic-segmentation` : Use classic method to segment histograms to extract relevant localisations
   #. :ref:`cellpose-segmentation` : Use Cellpose method to segment histograms to extract relevant localisations
   #. :ref:`ilastik-segmentation` : Use Ilastik method to segment histograms to extract relevant localisations

#. :ref:`membrane-performance` : Performance metrics calculation based on the localisations (not the histograms!)

.. note::

   This project is under active development.

For GitHub please see `GitHub <"https://github.com/oubino/locpix">`
For the bug tracker please see `Bug Tracker <"https://github.com/oubino/locpix/issues">`

.. toctree::
   :maxdepth: 1
   :hidden:

   User guide <user_guide/index>

   development/index

   API <_autosummary/locpix>

   release_notes
