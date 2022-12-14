######
locpix
######

**locpix** is a Python library for analysing point cloud data from SMLM.
This includes the following functionality:

#. Converting .csv files representing SMLM data (point cloud) into histograms :ref:`preprocess`
#. Manually annotating these histograms to extract relevant localisations :ref:`annotate`
#. Labelling histogram with seeds for watershed algorithm :ref:`get-markers`
#. Utilising Classic method, Cellpose and Ilastik to segment the histograms to extract relevant localisations :ref:`classic-segmentation`, :ref:`cellpose-segmentation` and :ref:`ilastik-segmentation`
#. Performance metrics calculation based on the localisations (not the histograms!) :ref:`membrane-performance`

.. note::

   This project is under active development.

.. toctree::
   :maxdepth: 1
   :hidden:

   User guide <user_guide/index>

   development/index

   API <_autosummary/locpix>

   release_notes
