#!/bin/bash

python -m pip install bump2version
bump2version --new-version patch setup.py || exit 1
