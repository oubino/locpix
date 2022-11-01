## Model architecture

## Recipes

### Recipe 1

Preprocess
Annotate
Train

### Recipe 2

Preprocess
GT label generation
Train

### Preprocess

Navigate to folder then run

```
python recipes/preprocess.py
```

This takes in the .csv files and converts them to datastructures

It saves these as .parquet files with name, dimensions and channels
as metadata while the dataframe is saved as a dataframe

The dataframe has the following columns:

x, y, z, channel, frame

Current limitations:
    Currently there is no option to manually choose which channels to consider, so all channels
    are considered.
    Drop zero label is set to False by default no option to change
    Drop pixel col is set to False by default no option to change

### Annotate or ...

This takes in the .parquet file, and allow the user to visualise in histogram
Then annotate - thus returning localisation level labels

These are added in a separate column to the dataframe called 'gt_label'

The dataframe is saved to parquet file with metadata specifying the mapping from 
label to integer

### ... GT label (alternative to annotate)

This first checks there isn't a column called gt_label - if there is it won't work

If there is not then it will create a new column called gt_label which will
be valued according to user specification

Save to parquet file metadata specifying the mapping from 
label to integer

Output the present labels and mapping as sanity check


### Pytorch geometric

Currently the location is taken in as feature vector i.e. the values of x and y
Obviously may want to play with this - number of photons etc.