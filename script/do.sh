#!/bin/bash
# Run another script inside a Python virtualenv.
# $1: other script to run
# $2: Python binary to use for virtualenv
set -u -o xtrace

# default to local Python 3.8 installation; use argument when building in GitHub runner
pyexe=${2:-/usr/local/bin/python3.8}
$pyexe -m pip install virtualenv
$pyexe -m virtualenv venv || exit 1
source venv/bin/activate
# shellcheck source=/dev/null.
source "$1"
deactivate || exit 1
rm -rf venv

set +o xtrace
