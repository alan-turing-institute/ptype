#!/bin/bash -e
# Build Python package into dist folder, and then test. Run from package root.
rm -rf dist # clean

python -m pip install -r requirements.txt
python -m pip freeze # useful for debugging

python setup.py sdist bdist_wheel

rm -rf build # temp dir used by setuptools (I think)
echo Built successfully.

# Test Python package found in dist folder.
python -m pip install dist/*.whl

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
python tests/test_ptype.py
compare_test_output column_type_counts.csv
compare_test_output column_type_predictions.json

echo Tests passed.
