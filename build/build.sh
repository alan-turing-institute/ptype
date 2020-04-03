#!/bin/bash
# Build Python package. Make this more repeatable/platform-independent as time goes on.

# doesn't seem to work if I run in a virtualenv (can't find 'src' module)
pyexe=/usr/local/bin/python3.8

# build source distribution
$pyexe ../setup.py sdist

# test
pushd ..
$pyexe tests/test_ptype.py
popd ... || exit
# python notebook_runner.py "../notebooks/demo.ipynb"
