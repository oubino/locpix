Development
===========

Git
---

If have accidentally committed to wrong branch need these series of commands

.. code-block:: console

   # Note: Any changes not committed will be lost.
   git branch newbranch # Create a new branch, saving the desired commits
   git checkout master # checkout master, this is the place you want to go back
   git reset --hard HEAD~3 # Move master back by 3 commits (Make sure you know how many commits you need to go back)
   git checkout newbranch # Go to the new branch that still has the desired commits


Docstring coverage
------------------

.. code-block:: console

   (locpix-env) $ pip install docstr-coverage
   (locpix-env) $ docstr-coverage src/locpix


Sphinx documentation
--------------------

Ran

.. code-block:: console

   (locpix-env) $ sphinx-quickstart docs

Choose yes for separate docs and src directories.

Then followed: `sphinx auto summary <https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough/62613202#62613202>`_

Before running

.. code-block:: console

   (locpix-env) $ make clean
   (locpix-env) $ make html


YAML files
^^^^^^^^^^

Have yaml files with configuration - need to convert these to .rst

.. code-block:: console

   (locpix-env) $ pip install yaml2rst


Then for all the template files convert to .rst , add heading and put in docs/source

.. code-block:: console

   (locpix-env) $ yaml2rst src/locpix/templates/*.yaml docs/source/*.rst

To support CI, we add this to the makefile for sphinx

To generates templates run

.. code-block:: console

   (locpix-env) $ make clean_templates
   (locpix-env) $ make templates

Then can run

.. code-block:: console

   (locpix-env) $ make clean
   (locpix-env) $ make html

.. warning::

   You get an error when running

   .. code-block:: console

      (locpix-env) $ make templates

   This is because the makefile contains a catch all, so it will run make templates then trys to run
   templates into Sphinx but this doesn't work!

.. warning::

   Note that to get the templates.rst in the correct format we had to edit yaml2rst.
   The additions are on lines 46-48

   .. code-block:: python

      title = os.path.basename(infilename).removesuffix('.yaml')
      print(title, file=outfh)
      print('='*len(title), file=outfh)

   Note this will fail on github actions - so need to include these functions as part of this package - not rely on yaml2rst!


PyPI
----

See this documentation for how to integrate publishing to PyPI using GitHub actions.

`PyPI <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>`_

However, as I protected master branch this was causing issues therefore moved towards this instead

`GitHub PyPI <https://www.seanh.cc/2022/05/21/publishing-python-packages-from-github-actions/>`_

#. Removed version from pyproject.toml as setuptools_scm finds this - however Sphinx needs this - therefore follow `version <https://pypi.org/project/setuptools-scm/>`_
under usage from sphinx - requires adding to the docs/conf.py file
#. Went up to "You could stop here" - could later implement automatic version increasing

Linting
-------

In line with GitHub actions run the following, note the GitHub editor is 127 chars wide

Python syntax errors or undefined names

.. code-block:: console

      (locpix-env) $ flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics


Note that exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide

.. code-block:: console

      (locpix-env) $ flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics


GitHub
------

CI
^^

Master branch is protected therefore have to checkout new branch and then merge this instead.

Steps for CI

#. Commit changes to working_branch and push this up to GitHub, no actions will run
#. When happy create pull request - this will trigger tests to run - if tests successful then merge to master
#. When merges, will generate documentation for master branch also runs CI again (maybe remove the latter). Could do this by removing on push from CI (but need to check that when the tests run on a pull request are they running on the new branch or the merged branch?)
#. When happy can create release on master and will only run publish workflow

Skip actions
^^^^^^^^^^^^

When push can choose not to run actions by including string

.. code-block:: console

   [skip actions]

To publish to pypi needs a tag therefore do in sequence

.. code-block:: console

      git checkout -b <branch-name>

Make changes then run (if don't want to push to pypi)

.. code-block:: console

      git add -A
      git commit
      git push origin <branch-name>

Then to push to pypi have to just push tagged master branch, where tag must start with v

.. code-block:: console

      git checkout master
      git tag <tag-name>
      git push origin <tag-name>


Code coverage
-------------

.. code-block:: console

      (locpix-env) $ pip install pytest
      (locpix-env) $ pip install pytest-cov
      (locpix-env) $ pytest --cov=src tests/


Cellpose train
--------------

To train cellpose first need .npy files of imgs and labels

Therefore use src/locpix/scripts/img_seg/custom_train.py to convert all images to .npy

Then move images to folder with Fov1->6 in train and move masks into this folder as well
Fov 7,8,9,10 in test and move masks into this folder as well

Had to append name _masks to each of the masks

Then run

.. code-block:: console

      (locpix-env) $ python -m cellpose --train --dir ~/imgs/train/ --test_dir ~/imgs/test/ --pretrained_model LC1 --chan 0 --chan2 0 --learning_rate 0.1 --weight_decay 0.0001 --n_epochs 3
      (locpix-env) $ python -m cellpose --dir ~/imgs/test/ --pretrained_model model --chan 0 --chan2 0 --save_png


python -m cellpose --train --dir "C:\Users\olive\OneDrive - University of Leeds\Project\output\locpix_project\cellpose_train\train" --test_dir "C:\Users\olive\OneDrive - University of Leeds\Project\output\locpix_project\cellpose_train\test" --pretrained_model LC1 --chan 0 --chan2 0 --learning_rate 0.1 --weight_decay 0.0001 --n_epochs 10 --min_train_masks 1 --verbose

python -m cellpose --dir  "C:\Users\olive\OneDrive - University of Leeds\Project\output\locpix_project\cellpose_train\test" --pretrained_model  "C:\Users\olive\OneDrive - University of Leeds\Project\output\locpix_project\cellpose_train\train\models\cellpose_residual_on_style_on_concatenation_off_train_2022_12_20_14_56_59.377919" --chan 0 --chan2 0 --save_png
