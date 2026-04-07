#!/bin/bash

# load conda
source venv/bin/activate

cd src

clear
PYTHONPATH=src python -m trainning.moe_train

