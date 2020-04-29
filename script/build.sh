#!/bin/bash -e
# Build Python package into dist folder. Run from package root.
rm -rf dist # clean

python -m pip install -r requirements.txt
python -m pip freeze # useful for debugging

python setup.py sdist bdist_wheel

rm -rf build # temp dir used by setuptools (I think)
echo Build completed successfully.
