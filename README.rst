.. image:: https://github.com/alan-turing-institute/ptype/workflows/build-publish/badge.svg?branch=release
    :target: https://github.com/alan-turing-institute/ptype/actions?query=workflow%3Abuild-publish+branch%3Arelease
    :alt: build-publish on release

.. image:: https://github.com/alan-turing-institute/ptype/workflows/build/badge.svg?branch=develop
    :target: https://github.com/alan-turing-institute/ptype/actions?query=workflow%3Abuild+branch%3Adevelop
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
    :target: https://mybinder.org/v2/gh/alan-turing-institute/ptype/release?filepath=notebooks
    :alt: Binder

============
Introduction
============

.. sectnum::

.. contents::

ptype is a probabilistic approach to type inference which is the task of identifying the data type (e.g., Boolean, date, integer and string) of a given column of data.

Existing approaches often fail on type inference for messy datasets, as the task becomes challenging in the presence of missing data and anomalies. With ptype_, our goal is to develop a robust method that can deal with such entries.

.. figure:: https://raw.githubusercontent.com/alan-turing-institute/ptype/release/notes/motivation.png
    :width: 400

    Normal, missing and anomalous values are denoted by green, yellow and red, respectively in the right hand figure.

.. _ptype: https://link.springer.com/content/pdf/10.1007/s10618-020-00680-1.pdf

ptype is built upon Probabilistic Finite-State Machines (PFSMs) which are used to model known data types, missing and anomalous data. We combine PFSMs such that a data column can be annotated via probabilistic inference in the proposed model, i.e., given a column of data we can infer column type and rows with missing and anomalous values. In contrast to standard use of FSMs (e.g., regular expressions) that either accepts or rejects a given data value, PFSMs provide probabilities and therefore offers the advantage of generating weighted predictions even when a column of data is consistent with more than one type model.

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

.. _Binder: https://mybinder.org/v2/gh/alan-turing-institute/ptype/release?filepath=notebooks
