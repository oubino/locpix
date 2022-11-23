Analysis of histograms/images of the SMLM data. 
Including: Classical methods, Cellpose and Ilastik

Project organisation
--------------------

Fill in 

Prerequisites
----------

You will need anaconda/miniconda or mamba.

Setup
-----

For security reasons we use a .env file to store the path to the data, this is ignored by Git by default.

Therefore, you should add a file called .env to the top level (i.e. same level as License, Makefile, Readme.md,...).
In this file you need one line with this

```
RAW_DATA_PATH = path/to/data_folder
```

Note that your directory CANNOT HAVE SPACES IN THE NAME i.e. if your directory is named "data/my data folder/" it will not work - you should rename your directory on your computer to something like "data/my_data_folder/"

This will assume all your .csv files are in data_folder - note the paths are normally taken as relative to the .env file - so this may take some fiddling around to get it correct (alternatively copy your data_folder folder into /data then the path would be = data/data_folder)

Create a new environment and activate it

```
conda create -n heptapods_img python==3.10
conda activate heptapods_img
```

Navigate to where you want this code repository.

Then clone this repository; move into it and install 

```
git clone git clone https://github.com/oubino/heptapods_img
cd heptapods_img
pip install .
```

Run tests 

```
pip install pytest
pytest -s
```

Running the code
----------------

All the code to run can be found in recipes.

Each recipe has a .yaml file 

This .yaml file specifies the recipe's configuration - and so should be amended before running each recipe!

Make sure you are located in the github directory

Preprocess the data
-------------------

```
python recipes/preprocess.py
```

Manually segment data
---------------------
```
python recipes/annotate.py
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
  conda install pytorch cudatoolkit=11.3 -c pytorch
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
