#!/bin/bash
set -u -o xtrace
# Build Python package. Run from package root.

# doesn't seem to work if I run in a virtualenv (can't find 'src' module)
# default to local Python 3.8 installation; use argument when building in GitHub runner.
pyexe=${1:-/usr/local/bin/python3.8}
$pyexe -m pip install virtualenv
$pyexe -m virtualenv venv || exit 1
source venv/bin/activate

python -m pip install -r requirements.txt
python -m pip freeze # useful for debugging
python setup.py sdist bdist_wheel || exit 1

compare_test_output () {
  if [[ $(git diff tests/$1) ]]
  then
    echo "Test failed."
    exit 1
  else
    echo "Test passed."
  fi
}

# seems to be included by default, except in GitHub runner or virtualenv
export PYTHONPATH=.
python tests/test_ptype.py || exit 1
compare_test_output column_type_counts.csv
compare_test_output column_type_predictions.json

deactivate || exit 1
rm -rf venv

set +o xtrace

rm -rf build # temp dir used by setuptools (I think)
echo Build completed successfully.

# python notebook_runner.py "../notebooks/demo.ipynb"
