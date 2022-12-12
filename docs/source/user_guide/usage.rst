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

Each script can be run with a GUI, but can also be run in headless mode.

In headless mode each script needs a configuration file (.yaml file), which should be 
specified using the -c flag.

Each configuration used, whether run in GUI or headless mode will be saved in the project directory.

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

To run the script using the GUI, run

.. code-block:: console

   (locpix-env) $ preprocess

To run the script without a GUI -i -c and -o flags should be specified

.. code-block:: console

   (locpix-env) $ preprocess -i path/to/input/data -c path/to/config/file -o path/to/project/directory

**API**
:py:mod:`locpix.scripts.preprocessing.preprocess`

.. _annotate:

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

.. _get-markers:

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

.. _classic-segmentation:

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

.. _cellpose-segmentation:

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

Save the project wherever with any name, but we recommend saving in this repository in folder 

.. code-block:: console
   
   models

with name

.. code-block:: console
   
   pixel_classification

Click the add new button under Raw Data.

Click add separate images.

Then navigate to 

.. code-block:: console
   
   data/ilastik/input_data 

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

We choose: Gaussian smoothing (3.5, 5); Laplacian of Gaussian (3.5, 5); 
Gaussian Gradient Magnitude (1.6, 3.5, 5); Difference of Gaussians (3.5, 5); 
Structure Tensor Eigenvalues (1.6, 3.5, 5, 10); Hessian of Gaussian 
Eigenvalues (3.5, 5, 10)

Then click training.

Click the label 1 and label the boundaries
Click the label 2 and label places inside cells
Note you can adjust brush size, rub it out etc.

After adding a few labels click live update - you could have clicked 
earlier as well.

Then keep adding manual labels until happy with the output - focus on 
areas it is predicting incorrectly.
i.e. Look at Prediction for label 1, and prediction for label 2. 
Click the eyes to turn off and on. Scroll the alphas to make more 
and less visible.

Then click on current view (LHS drop down) and change image.

We repeated this process for 5 images (Fov1,Fov2,Fov3,Fov5,Fov6), 
leaving the remaining 4 (Fov 7,8,9,10) to see what it would look on its own.
#TODO: #13 CHANGE THIS IF IMAGES CHANGE

Then click prediction export, make sure probabilities is chosen. 

Choose export image settings, choose format numpy.

.. code-block:: console
   
   .../smlm_analysis/data/output/ilastik/ilastik_pixel/npy/{nickname}.npy

where you should replace the ... with the path to your git repo.

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


Then run Ilastik 

Click new project: Boundary-based segmentation with Multicut.

We suggest naming this 

.. code-block:: console

   boundary_seg

and saving in 

.. code-block:: console
   
   models


Click under raw data add new and add separate images, 
now just add one image - we choose Fov1 - this will be located in 

.. code-block:: console

   data/ilastik/input_data

Then under probabilities add the corresponding probability output 
.npy file from previous stage 

This will be in

.. code-block:: console

   data/output/ilastik/ilastik_pixel/npy

 
N.B: make sure you click the add new button which is the higher of the two.

Then click DT Watershed. 

You can now mess with parameters and click update watershed until happy.

We used:
* Input channel: 0
* Threshold: 0.5
* Min boundary size: 0
* Presmooth: 3
* Seed labelling: Connected
* Min superpixel size: 100

Then click training and multicut. 

Then select features - I choose all features for raw data 0, 
probabilities 0 and 1.

Then left click to drop an edge right click to preserve an edge.

Then click live predict, then click update now to see updates to multicut.

Then click data export and choose same settings as before but now 
choose the dataset directory as

.. code-block:: console

   data/output/ilastik_boundary


i.e. the path will look like

.. code-block:: console

   .../smlm_analysis/data/output/ilastik/ilastik_boundary/npy/{nickname}.npy

Click Export all

Train/adjust just Fov1_DC 

Then do batch processing and select all remaining images and batch process

Then copied from ilastik to windows machine the output and 
put in ilastik_boundary
