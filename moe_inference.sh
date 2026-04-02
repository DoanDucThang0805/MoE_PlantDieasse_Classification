#!/bin/bash
# Installation script for the application
source venv/bin/activate

cd src
clear
PYTHONPATH=src python -m inference.moe.context_aware_moe_inference --use_context=True --router_mode="context_aware"
