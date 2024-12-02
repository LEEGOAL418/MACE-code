#!/bin/bash 
# during training, transform latest model from '.pt' format to '.model' format
# mace_run_train --config="./configs/LixC12.yaml" --max_num_epochs 0

model="MACE_model.model" # MACE_model_stage2.model
output_dir="tests/MACE_model"

test_file="../data/LixC12/dataset/train.extxyz"
mkdir -p $output_dir
output_file="${output_dir}/infer_train.extxyz"
mace_eval_configs \
    --configs=$test_file \
    --model=$model \
    --output=$output_file \
    --batch_size 16 \
    --device "cuda" \
    --default_dtype "float32"

test_file="../data/LixC12/dataset/test.extxyz"
mkdir -p $output_dir
output_file="${output_dir}/infer_test.extxyz"
mace_eval_configs \
    --configs=$test_file \
    --model=$model \
    --output=$output_file \
    --batch_size 1 \
    --device "cuda" \
    --default_dtype "float32"