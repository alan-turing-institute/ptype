#!/bin/bash
# $1: Python binary to use for virtualenv
# $2: PyPI password

script/do.sh "$1" script/bump-patch-version.sh
source do-build-test.sh "$1"
script/do.sh "$1" script/publish.sh "$2"
