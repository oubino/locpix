cellpose_train
==============
device to train on (gpu or cpu)
::

  gpu: True


model parameters
::

  model: pointnet


optimiser parameters
::

  optimiser: adam
  lr: 0.001
  weight_decay: 0.0001


training parameters
::

  epochs: 2
  batch_size: 1
  num_workers: 1 # generall higher -> faster
  loss_fn: nll

  train_files:
  - Fov1_DC
  - Fov2_DC
  - Fov3_DC
  - Fov5_DC
  - Fov6_DC

  test_files:
  - Fov7_DC
  - Fov8_DC
  - Fov9_DC
  - Fov10_DC
