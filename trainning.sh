#!/bin/bash

# load conda
source venv/bin/activate

cd src

# PYTHONPATH=src python -m trainning.mobilenetv2_train
clear
PYTHONPATH=src python -m trainning.vit_train
clear
PYTHONPATH=src python -m trainning.vit_train
clear
PYTHONPATH=src python -m trainning.vit_train

# PYTHONPATH=src python -m trainning.mobilenetv3_small_train

# PYTHONPATH=src python -m trainning.moe_train
