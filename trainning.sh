#!/bin/bash

# load conda
source venv/bin/activate

cd src

# PYTHONPATH=src python -m trainning.mobilenetv2_train
clear
PYTHONPATH=src python -m trainning.context_moe_train --router_mode="context_aware" --use_context=True




# PYTHONPATH=src python -m trainning.mobilenetv3_small_train

# PYTHONPATH=src python -m trainning.moe_train
