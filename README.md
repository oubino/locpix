Overview
--------

This package/repository provides a pipeline for analysing SMLM data.

This includes:
1. Converting .csv files representing SMLM data (point cloud) into histograms
2. Manually annotating these histograms to extract relevant localisations
3. Utilising Classic method, Cellpose and Ilastik to segment the histograms to extract relevant localisations
4. Performance metrics calculation based on the localisations (not the histograms!)

Project organisation
--------------------

To first use this code you will need the prerequisites detailed below, see "Prerequisites".

Once these are satisfied you will need to "Setup" detailed below.

To run the code see "Running the Code" below.

If you are

1. a novice coder, please follow "Beginner" steps.
2. an experienced coder, please follow "Advanced" steps.
3. developing the code, please follow "Developer" steps.

Please note that the difference between "Beginner" and "Advanced", is that in "Beginner" we use gui whereas in advanced we run in headless mode.
By running in headless mode, we can often be quicker to set up the code and allows for running on a hpc. 

All the code to run analysis can be found in src/locpix/scripts while all the library code is in the other folders under src/locpix


Directory structure
--------------------

Fill in 

Prerequisites
----------

### Beginner
-------------------

If you do not have anaconda installed please go to https://www.anaconda.com/ and install anaconda.

If you already have anaconda installed move on to setup!

Note this will be different if you are using linux, we make the potentially optimistic assumption that linux users will likely not need further details, but if you are a "Beginner" linux user please send a message or raise an issue and we will insert some instructions for linux users.

### Advanced
-------------------

You will need anaconda/miniconda or mamba.

We recommend mamba, though take care as mamba recommends fresh install without anaaconda.

Setup
-----

### Beginner
-------------------

Anaconda helps keep all the python packages and code in one contained environment.
We therefore create this environment (called locpix-env) and install python 3.10 in it.
We then activate this environment (we can have multiple environments therefore we need to make sure to activate the one we want to be in)
We can then install the locpixc code in this environment

To do this we need to open up Anaconda powershell, we can do this by searching on our computer for this application.

Type into the search bar "Anaconda powershell" and open it.

A terminal should appear.

In this terminal we need to navigate to where we want to do our work.
We don't strictly need to be in this folder to proceed, but it will make our lives easier late.
Therefore, we should cd to the working directory we want to be in to do the analysis.

If this is unfamiliar to you please watch https://www.youtube.com/watch?v=TQqJD-v6glE

Assume for example on my computer I have folder working_directory under C:\Users\Louise and my current working directory is C:\Users\Louise I would write in my Anaconda powershell

```
cd working_directory
```

Then enter the following three commands in the Anaconda powershell

```
conda create -n locpix-env python==3.10
conda activate locpix-env
pip install locpix
```

### Advanced
-------------------

You can either pip install the code inside a fresh environment as detailed above or install it inside a pre-existing environment.

Alternatively you can clone this repository and install it.

To do this we first need to change directory to the folder we want to be in (cd into the required directory in your terminal).
Then clone the repository

```
git clone https://github.com/oubino/locpix
```

Then move into the directory and install it

```
cd locpix
pip install .
```

We should then check the install was successful by running tests

```
pip install pytest
pytest -s
```

### Developer
-------------------

To develop the code clone this repository and we recommend installing an editable version, using the follow commands

```
git clone https://github.com/oubino/locpix
```

Then move into the directory and install it using an editable version

```
cd locpix
pip install -e .
```

We should then check the install was successful by running tests

```
pip install pytest
pytest -s
```

Running the code
----------------

### Basics

We provide an overview of the parts of the analysis

- Preprocess: convert .csv files containing localisations to .parquet files and wrangle into our data format
- Annotate: manually segment the localisations
- Get markers: Label each localisation image with a marker representing the cell
- Classic: Perform classic segmentation on our localisation dataset
- Cellpose: Perform cellpose segmentation on our localisation dataset
- Ilastik prep: Prepare data for Ilastik
- Ilastik output: Convert output of Ilastik into our format
- Membrane performance: Evaluate performance of membrane segmentation


#### Beginner
-------------------

To run analysis we need to define the configuration for each part of our analysis, when we do this we will save this configuration, in case we want to use this later.

This configuration will be saved to a .yaml file.

Each part of the analysis has an associated python script and .yaml file 

#### Advanced
-------------------

The scripts to perform analysis can be found in

```
src/locpix/scripts
```

while the library code is located in the other folders in 

```
src/locpix
```

Each script has an associated .yaml file which defines the configuration for the script. 

### Preprocess the data

We need to preprocess the data for later use, therefore this script must be run first.

This script will take in .csv files, and convert them to .parquet files, while also wrangling the data into our data format.

#### Beginner
-------------------

In the anaconda powershell write

```
preprocess -g -cg
```

This will open a GUI which will first ask for the directory containing the .csv files for preprocessing.

Then a GUI to set the configuration for preprocessing

The outputs of preprocessing including the configuration .yaml file will be saved in 

```
output/preprocess
```

We can also load back in a configuration file by clicking the "Load yaml" button in the GUI.


#### Advanced
-------------------

We need to specify where the .csv files are, there are 3 ways of doing this and 3 associated flags to pass

1. -f flag: path to folder with csvs OR
2. -g flag: use file dialogue gui to get path OR
3. -e flag: .csvs file are in .env file

Note the ability to load in the csv path from a .env file is provided for convenience.
For security reasons we choose by default for git to ignore the .env file therefore you should add a file called .env to the top level (i.e. same level as License, Makefile, Readme.md,...).
In this file you need one line with this

```
RAW_DATA_PATH = path/to/data_folder
```

Note that your directory CANNOT HAVE SPACES IN THE NAME i.e. if your directory is named "data/my data folder/" it will not work - you should rename your directory on your computer to something like "data/my_data_folder/"

Note we assume our folder contains all your .csv files and note the paths are normally taken as relative to the .env file - so this may take some fiddling around to get it correct (alternatively copy your data_folder folder into /data then the path would be = data/data_folder)

We also need to specify the configuration.
To do this we use one of two flags

1. -c flag: location of the .yaml configuration file, which we can edit using our text editor OR
2. -cg flag: Get configuration using a GUI

Therefore, to run preprocessing with data in a folder called input/data/ and configuration at input/preprocess.yaml, we would write

```
preprocess -f input/data -c input/preprocess.yaml
```

Manually segment data
---------------------

If we want to manually segment our data and return the segmented localisations we need to run this script.


#### Beginner
--------------

Run

```
annotate -cg
```

Use the gui to determine the configuration

#### Advanced
-------------------

Run

```
annotate -c path/to/configuration/for/annotate/yaml/file
```

e.g. If our configuration .yaml is located at input/annotate.yaml we could edit it there then run

```
annotate -c input/annotate.yaml
```

Get markers
-----------

```
python recipes/img_seg/get_markers.py
```

Classic segmentation
--------------------

```
python recipes/img_seg/classic.py
```

Cellpose segmentation
---------------------

### Requirements

Need to activate extra requirements - these are big and not included in initial install.

Note that if you have a GPU this will speed this up.

If you:

- have a GPU
  ```
  pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu117
  pip install cellpose
  ```
- don't have a GPU
    ```
    pip install pytorch cellpose
    ```

### Running

```
python recipes/img_seg/cellpose_module.py
```

Ilastik segmentation
--------------------

### Prep data

```
python recipes/img_seg/ilastik_prep.py
```

### Ilastik GUI*

See below what to run

### Ilastik output

```
python recipes/img_seg/ilastik_output.py
```

Test performance
----------------

```
python recipes/img_seg/membrane_performance.py
```

Ilastik GUI*
------------

We need to install ilastik
Install binary from https://www.ilastik.org/download.html 

### Ilastik membrane segmentation

Open ilastik.

Create a new project: Pixel Classification.

Save the project wherever with any name, but we recommend saving in this repository in folder 

```
/models
```
 with name 
```
pixel_classification
```
Click the add new button under Raw Data.

Click add separate images.

Then navigate to data/ilastik/input_data and select all the files at once and click open.
The axes should say yxc, and the shape should be (x_bins, y_bins, number channels).

Now click feature selection on LHS.

Click select features.

Choose the ones you feel are relevant.

Our recommendation: go through each row choosing all the sigmas for a row; Then click okay; Then on left hand side click on the features e.g. (Gaussian smoothing sigma 0.3 then Gaussian smoothing sigma 0.7) and evaluate which ones you think are pulling out relevant features; Then click select features again and remove ones you thought weren't useful!

We choose: Gaussian smoothing (3.5, 5); Laplacian of Gaussian (3.5, 5); Gaussian Gradient Magnitude (1.6, 3.5, 5); Difference of Gaussians (3.5, 5); Structure Tensor Eigenvalues (1.6, 3.5, 5, 10); Hessian of Gaussian Eigenvalues (3.5, 5, 10)

Then click training.

Click the label 1 and label the boundaries
Click the label 2 and label places inside cells
Note you can adjust brush size, rub it out etc.

After adding a few labels click live update - you could have clicked earlier as well.

Then keep adding manual labels until happy with the output - focus on areas it is predicting incorrectly.
i.e. Look at Prediction for label 1, and prediction for label 2. Click the eyes to turn off and on. Scroll the alphas to make more and less visible.

Then click on current view (LHS drop down) and change image.

We repeated this process for 5 images (Fov1,Fov2,Fov3,Fov5,Fov6), leaving the remaining 4 (Fov 7,8,9,10) to see what it would look on its own.
#TODO: #13 CHANGE THIS IF IMAGES CHANGE

Then click prediction export, make sure probabilities is chosen. 

Choose export image settings, choose format numpy.

Then in file put 
```
.../smlm_analysis/data/output/ilastik/ilastik_pixel/npy/{nickname}.npy
```
where you should replace the ... with the path to your git repo.

Click ok then click export all.

Save project (Ctrl + S)

Then close.

Then run 

```
make ilastik_pixel_output
```
to convert output to .csv

### Ilastik cell segmentation (requires linux subsytem for windows)

Batch multicut doesn't work via windows. Therefore, do this step in wsl2

Note all data will be on windows machine, therefore all paths on wsl2 need to point to the folders on the windows machine

One can see https://learn.microsoft.com/en-us/windows/wsl/install for setup instructions

We now will need to install Ilastik into this linux wsl2 subsystem as per Ilastik's instructions

Once you have tar the file, we run

```
./run_ilastik.sh
```

Then run Ilastik 


Click new project: Boundary-based segmentation with Multicut.

We suggest naming this 

```
boundary_seg
```
and saving in 

```
models
```

Click under raw data add new and add separate images, now just add one image - we choose Fov1 - this will be located in 

```
data/ilastik/input_data
```

Then under probabilities add the corresponding probability output .npy file from previous stage 

This will be in
```
data/output/ilastik/ilastik_pixel/npy
```
 
N.B: make sure you click the add new button which is the higher of the two.

Then click DT Watershed. 

You can now mess with parameters and click update watershed until happy.

We used:
- Input channel: 0
- Threshold: 0.5
- Min boundary size: 0
- Presmooth: 3
- Seed labelling: Connected
- Min superpixel size: 100

Then click training and multicut. 

Then select features - I choose all features for raw data 0, probabilities 0 and 1.

Then left click to drop an edge right click to preserve an edge.

Then click live predict, then click update now to see updates to multicut.

Then click data export and choose same settings as before but now choose the dataset directory as

```
data/output/ilastik_boundary
```

i.e. the path will look like

``` 
.../smlm_analysis/data/output/ilastik/ilastik_boundary/npy/{nickname}.npy
```
Click Export all

Train/adjust just Fov1_DC 

Then do batch processing and select all remaining images and batch process

Then copied from ilastik to windows machine the output and put in ilastik_boundary

Performance metrics
-------------------

Following semantic-kitti and other point dataset, we use mIoU and overall accuracy:

IOU is the intersection-over-union 

$$\frac{TP}{TP+FP+FN}$$

mIOU is the mean intersection over union - this is averaged across the classes - note background is counted as a class e.g. if we have background and membrane we have two classes so 

$$mIOU = (IOU_{background} + IOU_{membrane})/2$$

and oACC (overall accuracy) is 

$$\frac{TP+TN}{TP + TN + FP + FN}$$

Note during testing we aggregate all the test data into one aggregated dataset e.g. if we have 2 test dataitems:

1. 100 localisation; 60 membrane (ground truth/GT);  40 background (GT) 
2. 500 localisations; 150 membrane (GT); 350 background (GT)

We create an aggregated test dataset
- 600 localisations; 210 membrane; 390 background

The alternative approach is to calculate oACC and mIOU for each image, then mean these values accross all the dataitems. 

We did not use this approach as each data item contains different number of localisations, data items with 5 localisations would have same weight as one with 5000000.

For more on mIOU see http://www.semantic-kitti.org/tasks.html#semseg) 

We also produce ROC curves and precision-recall curves - where the latter is usually favoured in cases of imbalanced datasets, which we have here.

Similarly to above, we aggregate all the localisations into one aggregated test dataset and evaluate the precision and recall for all of these localisations. 

## Development

Sphinx documentation steps

Ran

```
sphinx-quickstart docs
```

Choose yes for separate docs and src directories.

Then followed:

https://www.sphinx-doc.org/en/master/tutorial/automatic-doc-generation.html

Before running

```
make html
```
