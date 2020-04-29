#!/bin/bash -e
# Publish contents of dist folder to PyPI.
# $1: PyPI password

python -m pip install --upgrade twine==3.1.1
python -m twine upload -u __token__ -p "$1" -r testpypi dist/*
echo Uploaded to PyPI.
