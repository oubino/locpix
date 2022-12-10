Overview
--------

This package/repository provides a pipeline for analysing SMLM data.

This includes:
1. Converting .csv files representing SMLM data (point cloud) into histograms
2. Manually annotating these histograms to extract relevant localisations
3. Utilising Classic method, Cellpose and Ilastik to segment the histograms to extract relevant localisations
4. Performance metrics calculation based on the localisations (not the histograms!)

This is a short ReadMe just containing a QuickStart guide.

For more comprehensive documentation please see https://oubino.github.io/locpix/ 


Insallation
-----------

Create an environment and intall locpix

```
   (base) $ conda create -n locpix-env python==3.10
   (base) $ conda activate locpix-env
   (locpix-env) $ pip install locpix
```

For developers and advanced users you can instead clone the repository and install using, where the -e flag means all changes to the source code are updated without requiring reinstallation - it is not not necessary and should not be used if you want the installed code to be immutable!

```    
    (locpix-env) $ git clone https://github.com/oubino/locpix
    (locpix-env) $ pip install ./locpix -e
```

We should then check the install was successful by running tests

```
  (locpix-env) $ pip install pytest
  (locpix-env) $ pytest -s
```

Quickstart
----------

### Overview

- Classic: Perform classic segmentation on our localisation dataset
- Cellpose: Perform cellpose segmentation on our localisation dataset
- Ilastik prep: Prepare data for Ilastik
- Ilastik output: Convert output of Ilastik into our format
- Membrane performance: Evaluate performance of membrane segmentation

### 



