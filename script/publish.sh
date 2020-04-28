#!/bin/bash
# Publish contents of dist folder to PyPI.
# $1: PyPI password

python -m pip install --upgrade twine==3.1.1 || exit 1
python -m twine upload -u __token__ -p "$1" -r testpypi dist/* || exit 1
echo Uploaded to PyPI.
