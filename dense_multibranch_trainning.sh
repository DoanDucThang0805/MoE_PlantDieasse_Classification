#!/bin/bash

source venv1/bin/activate
cd src

python -m trainning.mobilenetv3_small_dense_multibranch_train --seed 42
python -m trainning.mobilenetv3_small_dense_multibranch_train --seed 44
python -m trainning.mobilenetv3_small_dense_multibranch_train --seed 45
