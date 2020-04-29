#!/bin/bash -e
# Run another script inside a Python virtualenv.
# $1: Python binary to use for virtualenv
# $2: other script to run
# $3+: arguments to pass to script
set -u -o xtrace

pyexe=$1
$pyexe -m pip install virtualenv
$pyexe -m virtualenv venv
source venv/bin/activate
shift
# shellcheck source=/dev/null.
source "$@"
deactivate
rm -rf venv

set +o xtrace
