#!/bin/bash

# load conda
source venv/bin/activate

cd src

# PYTHONPATH=src python -m trainning.mobilenetv2_train

clear
PYTHONPATH=src python -m trainning.mobilenetv3_small_train
clear
PYTHONPATH=src python -m trainning.mobilenetv3_small_train
clear
PYTHONPATH=src python -m trainning.mobilenetv3_small_train

clear
PYTHONPATH=src python -m trainning.squeezenet_train
clear
PYTHONPATH=src python -m trainning.squeezenet_train
clear
PYTHONPATH=src python -m trainning.squeezenet_train

clear
PYTHONPATH=src python -m trainning.efficientnetv2s_train
clear
PYTHONPATH=src python -m trainning.efficientnetv2s_train
clear
PYTHONPATH=src python -m trainning.efficientnetv2s_train

clear
PYTHONPATH=src python -m trainning.efficientnetv2m_train
clear
PYTHONPATH=src python -m trainning.efficientnetv2m_train
clear
PYTHONPATH=src python -m trainning.efficientnetv2m_train

clear
PYTHONPATH=src python -m trainning.vgg16_train
clear
PYTHONPATH=src python -m trainning.vgg16_train
clear
PYTHONPATH=src python -m trainning.vgg16_train
