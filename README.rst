locpix
======

**locpix** is a Python library for analysing point cloud data from SMLM.

   This project is under active development

This is a short ReadMe just containing a QuickStart guide.
For more comprehensive documentation please see https://oubino.github.io/locpix/ 

**locpix** includes the following functionality in order they are used in a normal workflow:

#. `Preprocess`_ : Initialises project and converts .csv files representing SMLM data (point cloud) into .parquet files, **necessary for this software**
#. `Annotate`_ : Generating histograms from the SMLM data and manually annotating these histograms to extract relevant localisations
#. `Get markers`_ : Labelling histogram with seeds for watershed algorithm
#. Segmentation:

   #. `Classic segmentation`_ : Use classic method to segment histograms to extract relevant localisations 
   #. `Cellpose segmentation`_ : Use Cellpose method to segment histograms to extract relevant localisations 
   #. `Ilastik segmentation`_ : Use Ilastik method to segment histograms to extract relevant localisations 

#. `Membrane performance`_ : Performance metrics calculation based on the localisations (not the histograms!)

Project Structure
-----------------

We assume your input SMLM data are .csv files.

This input data must first be preprocessed into a user chosen project directory, using the  `Preprocess`_ script. 
We strongly suggest this project directory is located outside the locpix folder.

The input and output of all further scripts will remain located inside the project directory, the input data folder
will not be accessed again!

Usage configuration
-------------------

Each script can be run with a GUI, but can also be run in headless mode.

In headless mode each script needs a configuration file (.yaml file), which should be 
specified using the -c flag.

Each configuration used, whether run in GUI or headless mode will be saved in the project directory.

The templates for the configuration files can be found in the `templates folder <https://github.com/oubino/locpix/tree/master/src/locpix/templates>`_.

Quickstart
==========

Installation
------------

Prerequisites
^^^^^^^^^^^^^

You will need anaconda or miniconda or mamba.
We recommend `mamba <https://mamba.readthedocs.io/en/latest/>`_ 


Install
^^^^^^^

Create an environment and install via pypi 

.. code-block:: console

   (base) $ conda create -n locpix-env python==3.10
   (base) $ conda activate locpix-env
   (locpix-env) $ pip install locpix


Preprocessing
-------------

Preprocess
^^^^^^^^^^

This script preprocesses the input .csv data for later use AND **must be run first**.

This script will take in .csv files, and convert them to .parquet files, 
while also wrangling the data into our data format.

To run the script using the GUI, run

.. code-block:: console

   (locpix-env) $ preprocess

To run the script without a GUI -i -c and -o flags should be specified

.. code-block:: console

   (locpix-env) $ preprocess -i path/to/input/data -c path/to/config/file -o path/to/project/directory

Annotate
^^^^^^^^

This script allows for manual segmentation of the localisations.

To run the script using the GUI, run

.. code-block:: console

   (locpix-env) $ annotate

To run the script without a GUI -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ annotate -i path/to/project/directory -c path/to/config/file

Image segmentation
------------------

Get markers
^^^^^^^^^^^

This script allows for labelling the localisation image with a marker to represent the cells.

To run the script using the GUI, run

.. code-block:: console

   (locpix-env) $ get_markers

To run the script without a GUI -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ get_markers -i path/to/project/directory -c path/to/config/file

Classic segmentation
^^^^^^^^^^^^^^^^^^^^

Perform classic segmentation on our localisation dataset.

To run the script using the GUI, run

.. code-block:: console

   (locpix-env) $ classic

To run the script without a GUI -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ classic -i path/to/project/directory -c path/to/config/file

Cellpose segmentation (no training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Need to activate extra requirements - these are big and not included in initial install.

   Note that if you have a GPU this will speed this up.

   If you:

   * have a GPU
   .. code-block:: console

      (locpix-env) $ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
      (locpix-env) $ pip install cellpose
   
   * don't have a GPU
   .. code-block:: console

      (locpix-env) $ pip install pytorch cellpose


Perform Cellpose segmentation on our localisation dataset.

To run the script using the GUI, run

.. code-block:: console

   (locpix-env) $ cellpose_eval

To run the script without a GUI -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ cellpose_eval -i path/to/project/directory -c path/to/config/file


Cellpose segmentation (training)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

   Need to activate extra requirements - these are big and not included in initial install.

   Note that if you have a GPU this will speed this up.

   If you:

   * have a GPU
   .. code-block:: console

      (locpix-env) $ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
      (locpix-env) $ pip install cellpose

   * don't have a GPU
   .. code-block:: console

      (locpix-env) $ pip install pytorch cellpose
   

Prepare data for training 

.. code-block:: console

   (locpix-env) $ cellpose_train_prep -i path/to/project/directory -c path/to/config/file

Train cellpose (using their scripts)

.. code-block:: console

   (locpix-env) $ python -m cellpose --train --dir path/to/project/directory/cellpose_train/train --test_dir path/to/project/directory/cellpose_train/test --pretrained_model LC1 --chan 0 --chan2 0 --learning_rate 0.1 --weight_decay 0.0001 --n_epochs 10 --min_train_masks 1 --verbose

Evaluate cellpose

.. code-block:: console

   (locpix-env) $ cellpose_eval -i path/to/project/directory -c path/to/config/file -u -o cellpose_train_eval


Ilastik segmentation
^^^^^^^^^^^^^^^^^^^^

Need to prepare the data for Ilastik segmentation

.. code-block:: console

   (locpix-env) $ ilastik_prep -i path/to/project/directory -c path/to/config/file

Then run the data through the Ilastik GUI, which needs to be installed from
`Ilastik <https://www.ilastik.org/download.html>`_  and to run it 
please see `usage:Ilastik GUI`_

Then convert the output of the Ilastik GUI back into our format

.. code-block:: console

   (locpix-env) $ ilastik_output -i path/to/project/directory -c path/to/config/file

Membrane performance
^^^^^^^^^^^^^^^^^^^^

Need to evaluate the performance of the membrane segmentation

.. code-block:: console

   (locpix-env) $ membrane_performance -i path/to/project/directory -c path/to/config/file
