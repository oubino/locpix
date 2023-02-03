Performance metrics
===================


Following semantic-kitti and other point dataset, we use
mIoU and overall accuracy:

IOU is the intersection-over-union

.. math::

   \frac{TP}{TP+FP+FN}

mIOU is the mean intersection over union - this is averaged across the
classes - note background is counted as a class e.g. if we have
background and membrane we have two classes so

.. math::

   mIOU = \frac{(IOU_{background} + IOU_{membrane})}{2}

and oACC (overall accuracy) is

.. math::

   \frac{TP+TN}{TP + TN + FP + FN}

Note during testing we aggregate all the test data into one
aggregated dataset e.g. if we have 2 test dataitems:

#. 100 localisation; 60 membrane (ground truth/GT);  40 background (GT)
#. 500 localisations; 150 membrane (GT); 350 background (GT)

We create an aggregated test dataset
- 600 localisations; 210 membrane; 390 background

The alternative approach is to calculate oACC and mIOU for each image,
then mean these values accross all the dataitems.

We did not use this approach as each data item contains different
number of localisations, data items with 5 localisations would
have same weight as one with 5000000.

For more on mIOU see `semantic kitti <http://www.semantic-kitti.org/tasks.html#semseg>`_

We also produce ROC curves and precision-recall curves -
where the latter is usually favoured in cases of imbalanced datasets,
which we have here.

Similarly to above, we aggregate all the localisations into one
aggregated test dataset and evaluate the precision and recall
for all of these localisations.
