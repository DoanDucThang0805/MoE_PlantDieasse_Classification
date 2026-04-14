#!/bin/bash
# Installation script for the application
source venv/bin/activate

cd src
clear
PYTHONPATH=src python -m inference.pretrained.inference
