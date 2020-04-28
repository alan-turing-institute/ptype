#!/bin/bash

python -m pip install bump2version
bump2version patch || exit 1
