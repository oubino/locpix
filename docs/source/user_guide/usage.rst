Usage
=====

Project Structure
-----------------

We assume your input SMLM data are .csv files.

This input data must first be preprocessed into a user chosen project directory, using the `Preprocess`_ script.
We strongly suggest this project directory is located outside the locpix folder.

The input and output of all further scripts will remain located inside the project directory, the input data folder
will not be accessed again!

Usage configuration
-------------------

Each script needs a configuration file (.yaml file), which should be
specified using the -c flag.

Each configuration used will be saved in the project directory.

The templates for the configuration files can be found in :ref:`templates`

Preprocessing
-------------

There are two preprocessing scripts

* Preprocess
* Annotate

**API**
:py:mod:`locpix.scripts.preprocessing`

.. _preprocess:

Preprocess
^^^^^^^^^^

This script preprocesses the input .csv data for later use AND **must be run first**.

This script will take in .csv files, and convert them to .parquet files,
while also wrangling the data into our data format.

To run the script -i -c and -o flags should be specified

.. code-block:: console

   (locpix-env) $ preprocess -i path/to/input/data -c path/to/config/file -o path/to/project/directory

**API**
:py:mod:`locpix.scripts.preprocessing.preprocess`

.. _annotate:

Annotate
^^^^^^^^

This script allows for manual segmentation of the localisations.

To run the script -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ annotate -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.preprocessing.annotate`

Image segmentation
------------------

.. _get-markers:

Get markers
^^^^^^^^^^^

This script allows for labelling the localisation image with a marker to represent the cells.

To run the script -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ get_markers -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.img_seg.get_markers`

.. _classic-segmentation:

Classic segmentation
^^^^^^^^^^^^^^^^^^^^

Perform classic segmentation on our localisation dataset.

To run the script -i and -c flags should be specified

.. code-block:: console

   (locpix-env) $ classic -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.img_seg.classic`

.. _cellpose-segmentation:

Cellpose segmentation
^^^^^^^^^^^^^^^^^^^^^

.. warning::
   Need to activate extra requirements - these are big and not included in initial install.

   Note that if you have a GPU this will speed this up.

   Note we modified Cellpose to fit in with our analysis, therefore you need to install our forked repository - note below will clone the Cellpose repository to wherever you are located

   If you:

   * have a GPU

   .. code-block:: console

      (locpix-env) $ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
      (locpix-env) $ git clone https://github.com/oubino/cellpose
      (locpix-env) $ cd cellpose
      (locpix-env) $ pip install .

   * don't have a GPU

   .. code-block:: console

      (locpix-env) $ pip install pytorch
      (locpix-env) $ git clone https://github.com/oubino/cellpose
      (locpix-env) $ cd cellpose
      (locpix-env) $ pip install .

Perform Cellpose segmentation on our without any retraining on your dataset run the script with -i and -c flags specified

   .. code-block:: console

      (locpix-env) $ cellpose_eval -i path/to/project/directory -c path/to/config/file

To retrain first then evaluate we instead

   Prepare data for training
   Crucially this is also where the train/val/test split is defined and saved to the project metadata.

   .. code-block:: console

      (locpix-env) $ train_prep -i path/to/project/directory -c path/to/config/file

   Train cellpose

   .. code-block:: console

      (locpix-env) $ cellpose_train -i path/to/project/directory -ct path/to/config/train_file -ce path/to/config/eval_file


**API**
:py:mod:`locpix.scripts.img_seg.train_prep`
:py:mod:`locpix.scripts.img_seg.cellpose_eval`
:py:mod:`locpix.scripts.img_seg.cellpose_train`

.. _ilastik-segmentation:

Ilastik segmentation
^^^^^^^^^^^^^^^^^^^^

Need to prepare the data for Ilastik segmentation

.. code-block:: console

   (locpix-env) $ ilastik_prep -i path/to/project/directory -c path/to/config/file

Then run the data through the Ilastik GUI, please see `Ilastik GUI`_

Then convert the output of the Ilastik GUI back into our format

.. code-block:: console

   (locpix-env) $ ilastik_output -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.img_seg.ilastik_prep`
:py:mod:`locpix.scripts.img_seg.ilastik_output`

.. _membrane-performance:

Membrane performance
^^^^^^^^^^^^^^^^^^^^

Need to evaluate the performance of the membrane segmentation

.. code-block:: console

   (locpix-env) $ membrane_performance -i path/to/project/directory -c path/to/config/file

**API**
:py:mod:`locpix.scripts.img_seg.membrane_performance`

.. _ilastik-gui:

Ilastik GUI
^^^^^^^^^^^

We need to install ilastik
Install binary from `Ilastik <https://www.ilastik.org/download.html>`_

**Ilastik membrane segmentation**

Open Ilastik.

Create a new project: Pixel Classification.

Save the project wherever with any name, but we recommend saving in

.. code-block:: console

   path/to/project/directory/ilastik/models

you will have to create a new folder called models and save the project with name

.. code-block:: console

   pixel_classification

Click the add new button under Raw Data.

Click add separate images.

Note we are going to be loading in train images to train on then validation images to evaluate on for each fold

Then navigate to

.. code-block:: console

   path/to/project/directory/ilastik/prep

and select all the files at once and click open.
The axes should say yxc, and the shape should be (x_bins, y_bins, number channels).

Now click feature selection on LHS.

Click select features.

Choose the ones you feel are relevant.

Our recommendation: go through each row choosing all the sigmas for a row;
Then click okay; Then on left hand side click on the features
e.g. (Gaussian smoothing sigma 0.3 then Gaussian smoothing sigma 0.7)
and evaluate which ones you think are pulling out relevant features;
Then click select features again and remove ones you thought weren't useful!

We choose: All features.

Then click training.

Then for training images we loaded in the labels by right clicking on labels and choosing import.

For validation and test will make predictions.

Use print folds script to get files to train on for each fold.

Then click prediction export, make sure probabilities is chosen.

Choose export image settings, choose format numpy.

Choose file name

.. code-block:: console

   path/to/project/directory/ilastik/ilastik_pixel/{fold}/{nickname}.npy


Click ok then click export all.

Save project (Ctrl + S)

Then close.

**Ilastik cell segmentation (requires linux subsytem for windows)**

Batch multicut doesn't work via windows. Therefore, do this step in wsl2

Note all data will be on windows machine, therefore all paths on wsl2
need to point to the folders on the windows machine

One can see `wsl subsystem <https://learn.microsoft.com/en-us/windows/wsl/install>`_
for setup instructions

We now will need to install Ilastik into this linux wsl2 subsystem
as per Ilastik's instructions

Once you have tar the file, we run

.. code-block:: console

   (locpix-env) $ ./run_ilastik.sh

which will run Ilastik.

Click new project: Boundary-based segmentation with Multicut.

We suggest naming this

.. code-block:: console

   boundary_seg

and saving in

.. code-block:: console

   path/to/project/directory/ilastik/models

Click under raw data add new and add separate images,
add all images and remember to only train on train images!

.. code-block:: console

   path/to/project/directory/ilastik/prep

Then under probabilities add the corresponding probability output
.npy file from previous stage

This will be in

.. code-block:: console

   path/to/project/directory/ilastik/ilastik_pixel/npy

N.B: make sure you click the add new button which is the higher of the two.

Then click DT Watershed.

You can now mess with parameters and click update watershed until happy.

We used:
* Input channel: 0
* Threshold: 0.5
* Min boundary size: 0
* Smooth: 3
* Alpha: .9

Then click training and multicut.

Then select features - I choose all features.

Then left click to drop an edge right click to preserve an edge.

Then click then click update now to see updates to multicut.

View multicut edges and superpixel edges and correct the mistakes for each image.

Then click data export and choose same settings as before but now
choose the dataset directory as

.. code-block:: console

   path/to/project/directory/ilastik/ilastik_boundary

i.e. the path will look like

.. code-block:: console

   path/to/project/directory/ilastik/ilastik_boundary/npy/{nickname}.npy

.. warning::

   As you are in wsl2 the path to project directory will be different

   It will be

   .. code-block:: console

      /mnt/path/to/project/directory/ilastik/ilastik_boundary/npy/{nickname}.npy


      where the exact number of ../ at the beginning will depend on how deeply nested you are in the wsl.

      Further, you must ensure the slashes are forward not backward slashes.

      This may take time to get right, you may also have to put parts of the path in quotation marks

      Alternatively use their folder select function

Then click export all

Then save project : ctrl + s
