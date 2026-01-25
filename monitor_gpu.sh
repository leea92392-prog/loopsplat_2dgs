#!/bin/bash

LOG_FILE="gpu_memory.log"

# 写入表头
echo "timestamp,memory.used [MiB],memory.total [MiB]" > "$LOG_FILE"

while true; do
    # 获取当前时间戳和显存使用（单位 MiB）
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)

    # 写入日志
    echo "$TIMESTAMP,$USED,$TOTAL" >> "$LOG_FILE"

    sleep 1
done