#!/bin/bash
# Build Python package. Make this more repeatable/platform-independent as time goes on.

# run in virtualenv to avoid unwanted interactions with e.g. MacPorts which installs with sudo
python3 -m virtualenv venv
source ./venv/bin/activate

# build source distribution
python ../setup.py sdist

# more idiomatic way to achieve this?
export PYTHONPATH="src;test"

# test
pushd ..
python tests/test_ptype.py
popd
# python notebook_runner.py "../notebooks/demo.ipynb"

deactivate
rm -rf venv
