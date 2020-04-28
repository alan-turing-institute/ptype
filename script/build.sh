#!/bin/bash
# Build and test Python package. Run from package root.
# $1: Python binary to use for virtualenv
rm -rf dist # clean

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

rm -rf build # temp dir used by setuptools (I think)
echo Build completed successfully.
