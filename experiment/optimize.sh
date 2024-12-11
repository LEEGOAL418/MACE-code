#!/bin/bash

# 参数配置
MODEL_PATH="../experiment/MACE_model.model"
INIT_STRUCTURE_PATH="../structoptimize/initialcell/02_AB.extxyz"
OUTPUT_DIR="../structoptimize"  # 输出文件根目录
FMAX=0.01       # 收敛标准，单位eV/Å
OPTIMIZER="CG"  # 选择优化算法（CG 或 LBFGS）

# 调用优化程序
python ../mace4LiC/structural_optimize.py \
    --model_path $MODEL_PATH \
    --init_structure_path $INIT_STRUCTURE_PATH \
    --output_dir $OUTPUT_DIR \
    --optimizer $OPTIMIZER \
    --fmax $FMAX
