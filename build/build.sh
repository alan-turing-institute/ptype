#!/bin/bash
# Build Python package. Make this more repeatable/platform-independent as time goes on.

# run in virtualenv to avoid unwanted interactions with e.g. MacPorts which installs with sudo
python -m virtualenv venv
source ./venv/bin/activate

# build source distribution
python -m pip install --upgrade pip greenery jupyter_client ipykernel nbconvert nbformat setuptools wheel
python ../setup.py sdist

# test
python notebook_runner.py "../notebooks/demo.ipynb"

deactivate
rm -rf venv
