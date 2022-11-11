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
Run

```
python recipes/preprocess.py
```

Manually segment data
---------------------
Run
```
python recipes/annotate.py
```

Get markers
-----------

Run

```
python recipes/img_seg/get_markers.py
```

Classic segmentation
--------------------

Run

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
Run

```
python recipes/cellpose.py
```

Ilastik segmentation
--------------------






