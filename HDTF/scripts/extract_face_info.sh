#!/bin/bash

# ==== 可配置参数 ====
INPUT_DIR="/shareddisk/yexin/huanghj/data/TalkVid-160h/videos-crop"
OUTPUT_DIR="/shareddisk/yexin/huanghj/data/TalkVid-160h/new_face_info"
MODEL_ROOT="./model_ckpts/insightface_models"
NUM_GPUS=6
NUM_WORKERS=8
VIDEOS_PER_SHARD=600

# ==== 创建输出目录 ====
mkdir -p "$OUTPUT_DIR"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ==== 启动每张卡的任务 ====
for (( i=0; i<$NUM_GPUS; i++ ))
do
    echo "🚀 Launching GPU $i ..."
    CUDA_VISIBLE_DEVICES=$i python HDTF/scripts/extract_face_info_p.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_root "$MODEL_ROOT" \
        --gpu_id $i \
        --num_workers $NUM_WORKERS \
        --videos_per_shard $VIDEOS_PER_SHARD \
        > user/logs/extract_face_info/extract_face_info_$i.log 2>&1 &  # 每张卡输出日志
done

# ==== 等待所有任务完成 ====
wait
echo "✅ 所有任务执行完成"
