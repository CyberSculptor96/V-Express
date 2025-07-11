#!/bin/bash

# 输入目录（原始视频）
INPUT_DIR="/wangbenyou/huanghj/workspace/research/V-Express/HDTF/meta/resized_videos"

# 输出目录（采样结果）
OUTPUT_DIR="/wangbenyou/huanghj/workspace/V-Express/evaluation/datasets/HDTF/videos"
mkdir -p "$OUTPUT_DIR"

# 采样数量
NUM_SAMPLES=100

# 找出所有视频（可扩展为 avi、mov 等）
all_videos=($(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.mp4"))

# 检查视频数量是否足够
total=${#all_videos[@]}
if (( total < NUM_SAMPLES )); then
    echo "❌ 视频数量不足：仅找到 $total 个，要求 $NUM_SAMPLES 个"
    exit 1
fi

# 随机采样
sampled=($(printf "%s\n" "${all_videos[@]}" | shuf -n $NUM_SAMPLES))

# 拷贝
for file in "${sampled[@]}"; do
    cp "$file" "$OUTPUT_DIR/"
done

echo "✅ 已成功采样 $NUM_SAMPLES 条视频到：$OUTPUT_DIR"
