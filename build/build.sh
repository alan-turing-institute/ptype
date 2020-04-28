#!/bin/bash
set -u -o xtrace
# Build Python package.

# doesn't seem to work if I run in a virtualenv (can't find 'src' module)
# default to local Python 3.8 installation; use argument when building in GitHub runner.
pyexe=${1:-/usr/local/bin/python3.8}
$pyexe -m virtualenv venv
source venv/bin/activate

python -m pip install -r ../requirements.txt
pip freeze # useful for debugging

# build source distribution
python ../setup.py sdist || exit 1
python ../setup.py bdist_wheel || exit 1
rm -rf build # temp dir for bdist_wheel

compare_test_output () {
  if [[ $(git diff tests/$1) ]]
  then
    echo "Test failed."
    exit 1
  else
    echo "Test passed."
  fi
}

# test
pushd ..
  # seems to be included by default, except in GitHub runner or virtualenv
  export PYTHONPATH=.
  python tests/test_ptype.py || exit 1
  compare_test_output column_type_counts.csv
  compare_test_output column_type_predictions.json
popd ... || exit
deactivate
rm -rf venv

set +o xtrace
echo Build completed successfully.

# python notebook_runner.py "../notebooks/demo.ipynb"
