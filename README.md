![build-publish on release](https://github.com/alan-turing-institute/ptype-dmkd/workflows/build-publish/badge.svg?branch=release)
![build on develop](https://github.com/alan-turing-institute/ptype-dmkd/workflows/build/badge.svg?branch=develop)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/alan-turing-institute/ptype-dmkd/release?filepath=notebooks%2Fdemo.ipynb)

# Introduction
Type inference refers to the task of inferring the data type (e.g., Boolean, date, integer and string) of a given column of data, which becomes challenging in the presence of missing data and anomalies.

<p align="center">
  <img src="/notes/motivation.png" alt="drawing"  width="400"/>
</p>
<figcaption>Normal, missing and anomalous values are denoted by green, yellow and
red, respectively in the right hand figure.</figcaption>

[ptype](https://link.springer.com/content/pdf/10.1007/s10618-020-00680-1.pdf) is a probabilistic type inference model for tabular data, which aims to robustly infer the data type for each column in a table of data. By taking into account missing data and anomalies, ptype improves over the existing type inference methods. This repository provides an implementation of ptype in Python.

If you use this package, please cite ptype with the following BibTeX entry:
```bib
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
```


# Install requirements
```
pip install -r requirements.txt
```

# Usage

See demo notebooks in [`notebooks`] folder.
