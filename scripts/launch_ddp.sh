#!/bin/bash

# 设置默认值
GPUS=${1:-"5,6,7"}
NUM_GPUS=${2:-3}
MASTER_PORT=${3:-29500}

check_port_available() {
    if command -v nc &> /dev/null; then
        if nc -z localhost "$1" &> /dev/null; then
            echo "Error: Port $1 is already in use"
            return 1
        fi
    fi
    return 0
}

# 验证端口
if ! check_port_available "$MASTER_PORT"; then
    echo "Trying alternative port..."
    MASTER_PORT=$((MASTER_PORT + 1))
    if ! check_port_available "$MASTER_PORT"; then
        echo "Error: No available port found after trying $MASTER_PORT"
        exit 1
    fi
fi

echo "Launching training on GPUs: $GPUS"
echo "Using master port: $MASTER_PORT"

CUDA_VISIBLE_DEVICES=$GPUS uv run torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" scripts/pretrain.py