#!/bin/bash
# Build Python package. Make this more repeatable/platform-independent as time goes on.

# run in virtualenv to avoid unwanted interactions with e.g. MacPorts which installs with sudo
python -m virtualenv venv
source ./venv/bin/activate
python -m pip install --upgrade pip greenery nbformat setuptools wheel
python ../setup.py sdist
deactivate
rm -rf venv
