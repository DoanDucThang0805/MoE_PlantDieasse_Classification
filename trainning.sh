#!/bin/bash

# load conda
source venv/bin/activate

cd src

clear
python -m trainning.context_moe_train 
--batch_size 32 
--epochs 300 
--num_experts 6
--top_k 2
--model_name mobilenetv3large_moe 
--type_model MoE 
--router_mode context_aware 
--use_context True
