cellpose
========
Threshold/interpolation to apply to image
For interpolation options are: log2, log10 or linear
::

  img_threshold: 0
  img_interpolate: 'log2'


Model is trained on two-channel images, where the first channel is the channel to segment,
and the second channel is an optional nuclear channel."
Options for each:
  a. 0=grayscale, 1=red, 2=green, 3=blue
  b. 0=None (will set to zero), 1=red, 2=green, 3=blue
e.g. channels = [0,0] if you want to segment cells in grayscale
::

  channels: [0,0]


Files to evaluate Cellpose on
Options are: all (evaluate on all files), metadata (evaluate on files listed in metadata['test_files']) OR list of files to evaluate on
::

  test_files: all


Whether to use the GPU during evaluation
::

  use_gpu: True


The following is not generic, if you need to use this please raise an
issue and tag @oubino

Diameter to set for Cellpose model - NOTE: this makes no difference at the moment as we hardcode in the diameter
into our Cellpose fork, which needs to be changed!
::

  diameter: 100


Cellpose model - for a full list of models see https://cellpose.readthedocs.io/en/latest/models.html
::

  model: 'LC1'


For two channel image, we sum them together
Need the name of the first and second channel in terms of the real concepts
::

  channel: 'egfr'
  alt_channel: 'ereg'


Whether to sum the channels
::

  sum_chan: True
