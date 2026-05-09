#!/bin/bash

source venv1/bin/activate
cd src

python -m trainning.custom_mobilenetv3small_train --seed 42
python -m trainning.custom_mobilenetv3small_train --seed 43
python -m trainning.custom_mobilenetv3small_train --seed 44
python -m trainning.custom_mobilenetv3small_train --seed 45
python -m trainning.custom_mobilenetv3small_train --seed 46
