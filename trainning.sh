#!/bin/bash

# load conda
source venv/bin/activate

cd src

# PYTHONPATH=src python -m trainning.mobilenetv2_train

# PYTHONPATH=src python -m trainning.mobilenetv3_large_train

# PYTHONPATH=src python -m trainning.mobilenetv3_small_train

PYTHONPATH=src python -m trainning.mobilenetv4_train
