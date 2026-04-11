#!/bin/bash

# load conda
source ~/anaconda3/etc/profile.d/conda.sh

# activate environment
conda activate media/icnlab/Data/Thang/plan_dieases/env

cd src

clear
python -m trainning.mobilenetv3_small_train