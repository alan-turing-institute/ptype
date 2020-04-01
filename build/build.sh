#!/bin/bash
# Build Python package. Make this more repeatable/platform-independent as time goes on.

python -m pip install --upgrade pip greenery setuptools wheel
python ../setup.py sdist
