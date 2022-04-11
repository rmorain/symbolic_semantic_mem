#!/bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python experiments/00_baseline.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python experiments/01_description.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python experiments/02_max_attention.py
