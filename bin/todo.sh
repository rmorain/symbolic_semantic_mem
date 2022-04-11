#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python experiments/06_bert.py baseline 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python experiments/06_bert.py max_attention
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python experiments/06_bert.py median_attention
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python experiments/06_bert.py min_attention
