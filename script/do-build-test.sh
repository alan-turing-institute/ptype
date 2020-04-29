#!/bin/bash -e

script/do.sh "$1" script/build.sh
script/do.sh "$1" script/test.sh
