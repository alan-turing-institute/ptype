.. image:: https://github.com/alan-turing-institute/ptype-dmkd/workflows/build-publish/badge.svg?branch=release
    :target: https://github.com/alan-turing-institute/ptype-dmkd/actions?query=workflow%3Abuild-publish+branch%3Arelease
    :alt: build-publish on release

.. image:: https://github.com/alan-turing-institute/ptype-dmkd/workflows/build/badge.svg?branch=develop
    :target: https://github.com/alan-turing-institute/ptype-dmkd/actions?query=workflow%3Abuild+branch%3Adevelop
    :alt: build on develop

.. image:: https://badge.fury.io/py/ptype.svg
    :target: https://badge.fury.io/py/ptype
    :alt: PyPI version

.. image:: https://readthedocs.org/projects/ptype/badge/?version=latest
    :target: https://ptype.readthedocs.io/en/docs/index.html
    :alt: Documentation status

.. image:: https://pepy.tech/badge/ptype
    :target: https://pepy.tech/project/ptype
    :alt: Downloads

.. image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/alan-turing-institute/ptype-dmkd/release?filepath=notebooks
    :alt: Binder

============
Introduction
============

.. sectnum::

.. contents::

Type inference refers to the task of inferring the data type (e.g., Boolean, date, integer and string) of a given column of data, which becomes challenging in the presence of missing data and anomalies.

.. figure:: https://raw.githubusercontent.com/alan-turing-institute/ptype-dmkd/blob/release/notes/motivation.png
    :width: 400

    Normal, missing and anomalous values are denoted by green, yellow and red, respectively in the right hand figure.

ptype_ is a probabilistic type inference model for tabular data, which aims to robustly infer the data type for each column in a table of data. By taking into account missing data and anomalies, ptype improves over the existing type inference methods. This repository provides an implementation of ptype in Python.

.. _ptype: https://link.springer.com/content/pdf/10.1007/s10618-020-00680-1.pdf

If you use this package, please cite ptype with the following BibTeX entry:

::

    @article{ceritli2020ptype,
      title={ptype: probabilistic type inference},
      author={Ceritli, Taha and Williams, Christopher KI and Geddes, James},
      journal={Data Mining and Knowledge Discovery},
      year={2020},
      volume = {34},
      number = {3},
      pages={870–-904},
      doi = {10.1007/s10618-020-00680-1},
    }

====================
Install requirements
====================

.. code:: bash

    pip install -r requirements.txt

=====
Usage
=====

See demo notebooks in ``notebooks`` folder. View them online via Binder_.

.. _Binder: https://mybinder.org/v2/gh/alan-turing-institute/ptype-dmkd/release?filepath=notebooks
