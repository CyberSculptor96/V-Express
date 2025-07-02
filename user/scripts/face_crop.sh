#!/bin/bash
# ==== 检查是否传入 GPU ID ====
if [ -z "$1" ]; then
  echo "❌ 错误：你必须指定 GPU ID 作为第一个参数。例如："
  echo "    bash face_crop.sh 3"
  exit 1
fi

gpu_id=$1
export CUDA_VISIBLE_DEVICES=$gpu_id

# ==== 参数设置 ====
VIDEO_DIR="/shareddisk/yexin/huanghj/data/TalkVid-160h/videos"           # 输入视频目录（只包含.mp4）
OUTPUT_DIR="/shareddisk/yexin/huanghj/data/TalkVid-160h/videos-crop"         # 输出视频目录（将自动创建）
# VIDEO_DIR="data/test"           # 输入视频目录（只包含.mp4）
# OUTPUT_DIR="data/crop"         # 输出视频目录（将自动创建）
SIZE=512                                     # 裁剪后的视频分辨率（正方形）
NUM_WORKERS=16                                # 并行进程数
LOG="user/logs/face_crop_$gpu_id.2.log"

# ==== 创建输出目录 ====
mkdir -p "$OUTPUT_DIR"

# ==== 启动 Python 脚本 ====
python user/scripts/face_crop.py \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --size "$SIZE" \
    --num_workers "$NUM_WORKERS" \
    --gpu_id "$gpu_id" \
    > "$LOG" 2>&1 &
