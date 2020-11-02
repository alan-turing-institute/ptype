.. raw:: html

   <p align="center">
           <img width="60%" src="https://raw.githubusercontent.com/alan-turing-institute/CleverCSV/eea72549195e37bd4347d87fd82bc98be2f1383d/.logo.png">
           <br>
           <a href="https://travis-ci.org/alan-turing-institute/CleverCSV">
                   <img src="https://travis-ci.org/alan-turing-institute/CleverCSV.svg?branch=master" alt="Travis Build Status">
           </a>
           <a href="https://pypi.org/project/clevercsv/">
                   <img src="https://badge.fury.io/py/clevercsv.svg" alt="PyPI version">
           </a>
           <a href="https://clevercsv.readthedocs.io/en/latest/?badge=latest">
                   <img src="https://readthedocs.org/projects/clevercsv/badge/?version=latest" alt="Documentation Status">
           </a>
           <a href="https://pepy.tech/project/clevercsv">
                   <img src="https://pepy.tech/badge/clevercsv" alt="Downloads">
           </a>
           <a href="https://gitter.im/alan-turing-institute/CleverCSV">
                 <img src="https://badges.gitter.im/alan-turing-institute/CleverCSV.svg" alt="chat on gitter">
           </a>
           <a href="https://mybinder.org/v2/gh/alan-turing-institute/CleverCSVDemo/master?filepath=CSV_dialect_detection_with_CleverCSV.ipynb">
                   <img src="https://mybinder.org/badge_logo.svg" alt="Binder">
           </a>
           <a href="https://rdcu.be/bLVur">
                   <img src="https://img.shields.io/badge/DOI-10.1007%2Fs10618--019--00646--y-blue">
           </a>
   </p>

============
Introduction
============

.. sectnum::

.. contents::

Type inference refers to the task of inferring the data type (e.g., Boolean, date, integer and string) of a given column of data, which becomes challenging in the presence of missing data and anomalies.

.. figure:: ../notes/motivation.png
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

:: code:: bash

    pip install -r requirements.txt

=====
Usage
=====

See demo notebooks in ``notebooks`` folder.
