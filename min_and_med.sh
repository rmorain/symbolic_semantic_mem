#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python experiments/03_min_attention.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python experiments/05_median_attention.py
