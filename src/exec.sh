#!/bin/bash

python3 exec_eval.py --optimizer adam \
                     --batch_size 32 \
                     --epochs 10 \
                     --lr 5e-3 \
                     --epsilon 1e-7 \
                     --percent 0.1 \
                     --use_pretrained \
                     --use_lineval \
                     --downstream_path "/deep/group/img-graph/ViT-DRACON-1_FINAL" \
                     --model_type dracon \
                     --num_runs 5
