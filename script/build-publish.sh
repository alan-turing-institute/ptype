#!/bin/bash
# Build and publish a new patch version. Run from package root.
# $1: PyPI password

source script/bump-patch-version.sh
source script/build.sh
source script/publish.sh "$1"
