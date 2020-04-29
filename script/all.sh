#!/bin/bash
# Build and publish a new patch version. Run from package root.
# $1: PyPI password

script/do.sh script/bump-patch-version.sh
script/do.sh script/build.sh
script/do.sh script/test.sh
script/do.sh script/publish.sh "$1"
