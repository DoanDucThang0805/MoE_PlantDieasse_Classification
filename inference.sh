#!/bin/bash
# Installation script for the application
source venv/bin/activate

cd src
PYTHONPATH=src python -m inference.inference