classic
=======
Threshold/interpolation to apply to image
For interpolation options are: log2, log10 or linear
::

  img_threshold: 0
  img_interpolate: 'log2'


The following is not generic, if you need to use this please raise an
issue and tag @oubino

For two channel image, we visualise both channels then sum them together
Need the name of the first and second channel in terms of the real concepts

::

  channel: 'egfr'
  alt_channel: 'ereg'


Whether to sum the two channels
::

  sum_chan: True
