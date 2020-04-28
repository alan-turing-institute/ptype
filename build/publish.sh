#!/bin/bash
set -u -o xtrace
python3 -m twine upload -u __token__ -p "$1" -r testpypi dist/*
echo Uploaded to PyPI.
