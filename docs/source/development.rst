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

To support CI, we add this to the makefile for make htmls

Need to make this CI by:

#. Modify yaml2rst so that it formats the yaml files in way i want (including add heading)
#. When run make html - pip runs this and adds all to templates .rst

PyPI
----

`PyPI <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>`_

