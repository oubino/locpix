cellpose
========
threshold for visualisation
::

  vis_threshold: 5
  vis_interpolate: 'log2'


cellpose parameters
::

  model: 'LC1'
  diameter: 50

"Model is trained on two-channel images, where the first channel is the channel to segment,
and the second channel is an optional nuclear channel."
Options for each:
  a. 0=grayscale, 1=red, 2=green, 3=blue
  b. 0=None (will set to zero), 1=red, 2=green, 3=blue
e.g. channels = [0,0] if you want to segment cells in grayscale
::

  channels: [0,0]


whether to sum channels (currently only channel 0 and 1)
::

  sum_chan: False
