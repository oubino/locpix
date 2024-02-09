get_markers
===========
Threshold/interpolation to apply to image when visualising
For interpolation options are: log2, log10 or linear
::

  vis_threshold: 0
  vis_interpolate: 'log2'


The following is not generic, if you need to use this please raise an
issue and tag @oubino

For two channel image, we visualise both channels then sum them together
Need the name of the first and second channel in terms of the real concepts
::

  channel: 'egfr'
  alt_channel: 'ereg'
