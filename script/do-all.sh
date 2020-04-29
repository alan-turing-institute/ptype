#!/bin/bash
# Build and publish a new patch version. Run from package root.
# $1: Python binary to use for virtualenv
# $2: PyPI password

script/do.sh "$1" script/bump-patch-version.sh
script/do.sh "$1" script/build.sh
script/do.sh "$1" script/test.sh
script/do.sh "$1" script/publish.sh "$2"
