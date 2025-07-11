#!/bin/bash

# 输入视频目录
INPUT_DIR="/wangbenyou/huanghj/workspace/V-Express/evaluation/datasets/HDTF/videos"

# 输出视频目录
OUTPUT_DIR="/wangbenyou/huanghj/workspace/V-Express/evaluation/datasets/HDTF/videos_5s_24fps"
mkdir -p "$OUTPUT_DIR"

# 遍历所有 mp4 视频
for video in "$INPUT_DIR"/*.mp4; do
    filename=$(basename "$video")
    output="$OUTPUT_DIR/$filename"

    ffmpeg -y -i "$video" -t 5 -r 24 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p "$output" -loglevel error

    echo "✅ Processed $filename"
done
