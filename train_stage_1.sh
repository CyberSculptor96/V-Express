#!/bin/bash
export NO_ALBUMENTATIONS_UPDATE="1"

### ----- Configuration -----
export STAGE=3  # Training stage
export TRAIN="160h"  # Training dataset, can be "20h", "40h", "80h", or "160h"
export GPU="6"    # GPU ID to use, can be 0, 1, 2, etc.

config=./configs/$TRAIN/stage_$STAGE.yaml
zero_config=./configs/zero2_config.json
# zero_config=./training_configs/zero2_config_bf16.json

export CHIEF_IP=127.0.0.1
export CHIEF_PORT=21004     # port can not be the same!
export HOST_NUM=1
export INDEX=0

export HOST_GPU_NUM=1
export NCCL_IB_DISABLE=1
PROCESS_NUM=$((HOST_GPU_NUM * HOST_NUM))
echo "Total GPUS: ${PROCESS_NUM}"
echo "STAGE: ${STAGE}, TRAIN: ${TRAIN}, GPU id: ${GPU}"

LOG="./user/logs/train/stage_$STAGE-$TRAIN.log"

accelerate launch --gpu_ids $GPU --use_deepspeed --num_processes ${PROCESS_NUM} \
    --deepspeed_config_file $zero_config \
    --num_machines "${HOST_NUM}" --machine_rank "${INDEX}" --main_process_ip "${CHIEF_IP}" --main_process_port "${CHIEF_PORT}" \
    --deepspeed_multinode_launcher standard \
    train.py  --config $config \
    > $LOG 2>&1 &
