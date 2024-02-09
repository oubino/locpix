preprocess
==========
Name of the column which has the x, y, z (if present) coordinates,
the channel and frame information in the csvs being processed
[If not present put null]
::

  x_col: 'X (nm)'
  y_col: 'Y (nm)'
  z_col: null
  channel_col: 'Channel'
  frame_col: 'Frame'


The number of dimensions to consider
If 2 only deals with x and y
If 3 will read in and deal with z as well (currently not fully supported)
::

  dim: 2


Choice of which channels user wants to consider
::

  channel_choice: [0,1,2,3]


Label for what each channel represents in real terms where unk is unknown
::

  channel_label: ['egfr','ereg','unk','unk']


Files to include - if want all files
::

  include_files: all


Otherwise list the files without extensions (e.g. .csv)
include_files:
- file_1
- file_2
