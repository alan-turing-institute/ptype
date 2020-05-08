#!/bin/bash -e

# Build Python package into dist folder, and then test.
build () {
  rm -rf dist # clean

  python -m pip install -r requirements.txt
  python -m pip freeze # useful for debugging

  python setup.py sdist bdist_wheel

  rm -rf build # temp dir used by setuptools (I think)
  echo Built successfully.

  # Test Python package found in dist folder.
  install () {
    python -m pip install dist/*.whl
  }
}

compare_test_output () {
  if [[ $(git diff tests/$1) ]]
  then
    echo "Test failed."
    exit 1
  else
    echo "Test passed."
  fi
}

test () {
  # seems to be included by default, except in GitHub runner or virtualenv
  export PYTHONPATH=.
  time python tests/test_ptype.py
  compare_test_output column_type_counts.csv

  echo Tests passed.
}

build
install
test
