#!/bin/bash -e
# $1: Python binary to use for virtualenv
# $2: PyPI password
# $3: PyPI repository to use (pypi or testpypi)

script/do.sh "$1" script/bump-patch-version.sh
source script/do-build.sh "$1"
script/do.sh "$1" script/publish.sh "$2" "$3"
git commit -a -m "Bump version number."
git push
git push --tags
