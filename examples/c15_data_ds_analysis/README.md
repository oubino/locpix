## Downstream analysis on C15 data

This notebook shows some of the downstream analysis that can be applied to the segmentations.

We use the output from retrained Cellpose on the C15 dataset from fold 0.

Specifically

We copied folders

+ cellpose_train/0/cell/seg_dataframes
+ cellpose_train/0/cell/seg_img
+ membrane_performance/cellpose_train/membrane/seg_dataframes/test/0

The test files for this fold are 

C15_EREG568_EGFR647_FOV3.parquet
C15_EREG568_EGFR647_FOV5.parquet
C15_EREG647_EGFR568_FOV1.parquet
C15_EREG647_EGFR568_FOV5.parquet
C15_EREG647_EGFR568_FOV7.parquet
C15_EREG647_EGFR568_FOV8.parquet
C15_EREG647_EGFR568_FOV10.parquet
