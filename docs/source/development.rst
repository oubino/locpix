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


PyPI
----

`PyPI <https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/>`_

