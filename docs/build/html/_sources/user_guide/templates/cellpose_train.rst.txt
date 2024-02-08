cellpose_train
==============
Threshold/interpolation to apply to image
For interpolation options are: log2, log10 or linear
::

  img_interpolate: log2
  img_threshold: 0


Cellpose model we are training
::

  model: LC1


Learning rate/weight decay/epochs for training
::

  learning_rate: 0.01
  weight_decay: 0.0001
  epochs: 1000


Whether to use GPU for training
::

  use_gpu: True



The following is not generic, if you need to use this please raise an
issue and tag @oubino

For two channel image, we visualise both channels then sum them together
Need the name of the first and second channel in terms of the real concepts
In this case in the form of a list
::

  channels: ["egfr", "ereg"]


Whether to sum the two channels
::

  sum_chan: True
