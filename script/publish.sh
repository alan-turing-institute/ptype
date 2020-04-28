#!/bin/bash
# Publish contents of dist folder to PyPI.
# $2: PyPI password
# $1: Python binary to use for virtualenv
set -u -o xtrace

# Same pattern as build.sh for "reproducible" Python behaviour.
pyexe=${2:-/usr/local/bin/python3.8}
$pyexe -m pip install virtualenv
$pyexe -m virtualenv venv || exit 1
source venv/bin/activate

python -m pip install --upgrade twine==3.1.1 || exit 1
python -m pip install bump2version
bump2version patch setup.py
python -m twine upload -u __token__ -p "$1" -r testpypi dist/* || exit 1

deactivate || exit 1
rm -rf venv

set +o xtrace
echo Uploaded to PyPI.
