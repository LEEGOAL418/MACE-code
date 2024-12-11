#!/bin/bash

# 传入log文件路径作为参数
LOG_FILE_PATH=$1

# 检查是否传入log文件路径
if [ -z "$LOG_FILE_PATH" ]; then
    echo "Error: Log file path is required."
    exit 1
fi

# 运行Python脚本绘图
python3 ../mace4LiC/plot_loss.py "$LOG_FILE_PATH"
