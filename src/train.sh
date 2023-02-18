#!/bin/bash

mkdir -p /deep/u/sameerk/ImgGraph/dracon_train

python3 exec_train.py --graph_root "/deep/group/img-graph" \
                      --raw_image_path "/deep/group/data/med-data/cxr.h5" \
                      --processed_image_path "/deep/group/img-graph/MIMIC/img_with_attr.h5" \
                      --processed_graph_path "/deep/group/img-graph/MIMIC/graphs_with_attr.pt" \
                      --model_root "/deep/u/sameerk/ImgGraph/dracon_train" \
                      --optimizer adam \
                      --use_pool \
                      --batch_size 250 \
                      --epochs 10 \
                      --graph_transform reaction \
                      --image_freeze_interval 20 \
                      --lr 1e-3 \
                      --momentum 1e-3 \
                      --graph_architecture DRACON \
                      --attn_heads 2 \
                      --use_rgcn \
                      --trans_layers 1 \
                      --fc_layers 2 \
                      --graph_layers 2 \
                      --graph_hidden 512 \
                      --num_runs 10
