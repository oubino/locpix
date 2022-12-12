Development
===========

Sphinx documentation
--------------------

Ran

.. code-block:: console

   (locpix-env) $ sphinx-quickstart docs

Choose yes for separate docs and src directories.

Then followed: `sphinx auto doc <https://www.sphinx-doc.org/en/master/tutorial/automatic-doc-generation.html>`_

Before running

.. code-block:: console

   (locpix-env) $ make clean
   (locpix-env) $ make html

Needed to install sphinx-autoapi

.. code-block:: console

   (locpix-env) $ pip install sphinx-autoapi

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

Master branch is protected therefore have to checkout new branch and then merge this instead.

When push can choose not to run actions by including string