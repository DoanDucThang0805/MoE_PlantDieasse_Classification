#!/bin/bash
# Installation script for the application
source venv/bin/activate
source venv/Scripts/activate

cd src
clear
PYTHONPATH=src python -m inference.pretrained.inference
