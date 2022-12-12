preprocess
==========
The following are the names of the
x column, y column, z column if present, channel, frame,
in the csvs being processed
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


choice of which channels user wants to consider
::

  channel_choice: [0,1,2,3]


whether to not drop the column containing
pixel
::

  drop_pixel_col: False


files to include
::

  include_files:
  - Fov1_DC
  - Fov2_DC
  - Fov3_DC
  - Fov5_DC
  - Fov6_DC
  - Fov7_DC
  - Fov8_DC
  - Fov9_DC
  - Fov10_DC
