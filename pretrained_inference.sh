#!/bin/bash
# Installation script for the application
source venv1/bin/activate
clear
cd src
PYTHONPATH=src python -m inference.pretrained.inference
