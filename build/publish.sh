#!/bin/bash
# $2: TestPyPI password
# $1: Python binary to use for virtualenv
set -u -o xtrace

# Same pattern as build.sh for "reproducible" Python behaviour.
pyexe=${2:-/usr/local/bin/python3.8}
$pyexe -m virtualenv venv
source venv/bin/activate

python -m pip install --upgrade twine==3.1.1 || exit 1
python -m twine upload -u __token__ -p "$1" -r testpypi dist/* || exit 1

set +o xtrace
echo Uploaded to PyPI.
