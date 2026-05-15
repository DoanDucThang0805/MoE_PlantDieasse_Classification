#!/bin/bash

# load conda
source venv/bin/activate

cd src
clear

# PYTHONPATH=src python -m trainning.mobilenetv2_train

clear
PYTHONPATH=src python -m trainning.mobilenetv3_small_train --seed 42
PYTHONPATH=src python -m trainning.mobilenetv3_small_train --seed 43
PYTHONPATH=src python -m trainning.mobilenetv3_small_train --seed 44
PYTHONPATH=src python -m trainning.mobilenetv3_small_train --seed 45
PYTHONPATH=src python -m trainning.mobilenetv3_small_train --seed 46



