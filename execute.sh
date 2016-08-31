#!/bin/bash

# Exit on error
set -o errexit

jupyter nbconvert --inplace --execute --ExecutePreprocessor.timeout=-1 *.ipynb
jupyter nbconvert --to=script --FilesWriter.build_directory=scripts *.ipynb
