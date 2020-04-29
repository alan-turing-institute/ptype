#!/bin/bash
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
python tests/test_ptype.py || exit 1
compare_test_output column_type_counts.csv
compare_test_output column_type_predictions.json

echo Tests passed.
