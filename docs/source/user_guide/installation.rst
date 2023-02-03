Installation
============

Prerequisites
-------------

You will need anaconda or miniconda or mamba.

We recommend `mamba <https://mamba.readthedocs.io/en/latest/>`_

However, if you are a novice, anaconda is more ubiquitous and we would recommend this `anaconda <https://www.anaconda.com/>`_

Install
-------

Open up your terminal or anaconda powershell.
Then change directory to the directory you would like to store the code in.
If you are unfamiliar with how to do this please see `change directory help <https://www.youtube.com/watch?v=TQqJD-v6glE>`_

Then create an environment and install via pypi

.. code-block:: console

   (base) $ conda create -n locpix-env python==3.10
   (base) $ conda activate locpix-env
   (locpix-env) $ pip install locpix

.. note::

   For developers and advanced users you can instead clone the repository and install.
   Note the editable flag -e means any changes you make to the
   code are implemented without requiring re-installation - this is not necessary and
   should not be used if you want the installed code to be immutable!

   .. code-block:: console

      (locpix-env) $ git clone https://github.com/oubino/locpix
      (locpix-env) $ pip install ./locpix -e

We should then check the install was successful by running tests

.. code-block:: console

   (locpix-env) $ pip install pytest
   (locpix-env) $ pytest -s
