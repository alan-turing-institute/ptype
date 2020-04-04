#!/bin/bash
set -x
# Build Python package.

# doesn't seem to work if I run in a virtualenv (can't find 'src' module)
# default to local Python 3.8 installation; use argument when building in GitHub runner.
pyexe=${1:-/usr/local/bin/python3.8}

# build source distribution
$pyexe ../setup.py sdist || exit 1

# test
pushd ..

# this seems to be included by default, but not in the GitHub runner
PYTHONPATH=.

$pyexe tests/test_ptype.py || exit 1
# show disparities, then discard; will check these later
git diff
git checkout tests/column_type_counts.csv
git checkout tests/column_type_predictions.json

popd ... || exit

# python notebook_runner.py "../notebooks/demo.ipynb"
