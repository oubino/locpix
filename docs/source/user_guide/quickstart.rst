Quickstart
==========

Project Structure
-----------------

We assume your input SMLM data are .csv files.

This input data must first be preprocessed into a user chosen project directory, using the `Preprocess`_ script. 
We strongly suggest this project directory is located outside the locpix folder.

The input and output of all further scripts will remain located inside the project directory, the input data folder
will not be accessed again!

Usage configuration
-------------------

Each script can be run with a GUI, but can also be run in headless mode.

In headless mode each script needs a configuration file (.yaml file), which should be 
specified using the -c flag.

Each configuration used, whether run in GUI or headless mode will be saved in the project directory.

The templates for the configuration files can be found in :ref:`templates`

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

**API**
:py:mod:`locpix.scripts.preprocessing.preprocess`

Annotate
^^^^^^^^

This script allows for manual segmentation of the localisations.

To run the script using the GUI, run

.. code-block:: console

   (locpix-env) $ annotate

To run the script without a GUI -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ annotate -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.preprocessing.annotate`

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

**API**
:py:mod:`locpix.scripts.img_seg.get_markers`

Classic segmentation
^^^^^^^^^^^^^^^^^^^^

Perform classic segmentation on our localisation dataset.

To run the script using the GUI, run

.. code-block:: console

   (locpix-env) $ classic

To run the script without a GUI -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ classic -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.img_seg.classic`

Cellpose segmentation
^^^^^^^^^^^^^^^^^^^^^

.. warning::
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

   (locpix-env) $ cellpose

To run the script without a GUI -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ cellpose -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.img_seg.cellpose`

Ilastik segmentation
^^^^^^^^^^^^^^^^^^^^

Need to prepare the data for Ilastik segmentation

.. code-block:: console

   (locpix-env) $ ilastik_prep -i path/to/project/directory -c path/to/config/file

Then run the data through the Ilastik GUI, which needs to be installed from
`Ilastik <https://www.ilastik.org/download.html>`_  and to run it please see :ref:`ilastik-gui`.

Then convert the output of the Ilastik GUI back into our format

.. code-block:: console

   (locpix-env) $ ilastik_output -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.img_seg.ilastik_prep`
:py:mod:`locpix.scripts.img_seg.ilastik_output`


Membrane performance
^^^^^^^^^^^^^^^^^^^^

Need to evaluate the performance of the membrane segmentation

.. code-block:: console

   (locpix-env) $ membrane_performance -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.img_seg.membrane_performance`
