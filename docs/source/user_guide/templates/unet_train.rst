unet_train
==========

Training parameters including batch size, epochs, learning rate, weight decay, loss function (dice or bce), number of workers, whether to use the GPU
::

  batch_size: 1
  epochs: 1
  learning_rate: 0.01
  weight_decay: 0.0001
  loss_fn: bce
  num_workers: 1
  use_gpu: true

Threshold/interpolation to apply to image
For interpolation options are: log2, log10 or linear
::

  img_interpolate: log2
  img_threshold: 0

Transformations to apply during training/test
::

  train_transforms:
    dtypeconv: null
    erasing: null
    h_flip: null
    perspective: 0.2
    rotation: 90
    v_flip: null

  test_transforms:
    dtypeconv: null

Weights and bias parameters
::

  wandb_dataset: c15_imgs
  wandb_project: c15_img_unet_bce

The following is not generic, if you need to use this please raise an
issue and tag @oubino

For two channel image, we visualise both channels then sum them together
Need the name of the first and second channel in terms of the real concepts
::

  channels:
  - egfr
  - ereg

Whether to sum the two channels
::

  sum_chan: True
