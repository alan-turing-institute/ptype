#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import find_packages, setup

# Package meta-data.
AUTHOR = "Taha Ceritli, Christopher K. I. Williams, James Geddes, Roly Perera"
DESCRIPTION = "Probabilistic type inference"
EMAIL = "t.y.ceritli@sms.ed.ac.uk, ckiw@inf.ed.ac.uk, jgeddes@turing.ac.uk, rperera@turing.ac.uk"
LICENSE = "MIT"
LICENSE_TROVE = "License :: OSI Approved :: MIT License"
NAME = "ptype"
REQUIRES_PYTHON = ">=3.6.0"
URL = "https://github.com/alan-turing-institute/ptype"
VERSION = None

# What packages are required for this module to be executed?
REQUIRED = [
    "greenery>=3.2",
    "joblib>=0.17.0",
    "matplotlib>=3.3.0",
    "numpy>=1.19.0",
    "pandas>=1.1.0",
    "scikit-learn>=0.24.2"
    "scipy>=1.5.0",
]

docs_require = ["clevercsv>=0.6.0", "scikit-learn>=0.22.0"]
test_require = ["jsonpickle>=1.4.0", "nbval>=0.9.6"]
dev_require = ["wheel"]

# What packages are optional?
EXTRAS = {
    "docs": docs_require,
    "test": test_require,
    "dev": docs_require + test_require + dev_require,
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.rst"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

# Where the magic happens:
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/x-rst",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(
        exclude=["tests", "*.tests", "*.tests.*", "tests.*", ".DS_Store*"]
    ),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license=LICENSE,
    ext_modules=[],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        LICENSE_TROVE,
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
        "Topic :: Utilities",
    ],
)
