#!/bin/bash
# Build Python package. Make this more repeatable/platform-independent as time goes on.

# run in virtualenv to avoid unwanted interactions with e.g. MacPorts which installs with sudo
python3 -m virtualenv venv
source ./venv/bin/activate

# build source distribution
python3 -m pip install --upgrade pip greenery jupyter_client ipykernel nbconvert nbformat setuptools wheel
python3 ../setup.py sdist

# Presumably a better way to do this, although PyCharm does something similar.
python3 -c "import sys; sys.path.extend(['src', 'test'])"

# test
pushd ..
python3 tests/test_ptype.py
popd
# python notebook_runner.py "../notebooks/demo.ipynb"

deactivate
rm -rf venv
