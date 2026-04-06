#!/bin/bash

# load conda
source venv/bin/activate

cd src

clear
PYTHONPATH=src python -m trainning.vit_train
clear
PYTHONPATH=src python -m trainning.vit_train
