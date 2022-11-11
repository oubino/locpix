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

Navigate to where you want this code repository, clone this repository

```
git clone https://github.com/oubino/smlm_analysis
```
Create environment then activate it

If on:

- Linux: 
    ```
    make create_environment
    conda activate ENV_NAME
    ```
- Windows:
    ```
    conda create -n smlm_analysis python=3.9
    conda activate smlm_analysis
    ```

Install dependencies

```
make requirements
```

Install smlm - this will be depreceated once smlm is open source

```
pip install git+https://github.com/oubino/smlm.git
```

For remaining instructions, could simplify makefile on Linux but windows doesn't allow easy creation of directories from makefile - therefore default to more convoluted Windows way which should still work on Linux.

Furthermore, we trialed doing all on Linux subsystem for windows BUT napari doesn't like wsl2!

Preprocess the data
-------------------

Create directories
```
mkdir data/dataitems
mkdir data/dataitems/histo
```

Perform preprocessing
```
make preprocess
```

Manually segment data
---------------------

Create directories
```
mkdir data/seg/csvs
mkdir data/seg/histo
mkdir data/seg/histo_boundary
```

Perform manual segmentation

```
make manual_segment
```
