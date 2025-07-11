#!/bin/bash

# 输入视频目录
VIDEO_DIR="/wangbenyou/huanghj/workspace/V-Express/evaluation/datasets/HDTF/gt"

# 输出图像目录
IMAGE_DIR="/wangbenyou/huanghj/workspace/V-Express/evaluation/datasets/HDTF/imgs"
mkdir -p "$IMAGE_DIR"

# 输出音频目录
AUDIO_DIR="/wangbenyou/huanghj/workspace/V-Express/evaluation/datasets/HDTF/audios"
mkdir -p "$AUDIO_DIR"

# 遍历所有 mp4 视频
for video in "$VIDEO_DIR"/*.mp4; do
    filename=$(basename "$video" .mp4)

    # 提取首帧图像为 JPEG
    ffmpeg -y -i "$video" -vf "select=eq(n\,0)" -q:v 2 "$IMAGE_DIR/${filename}.jpg" -loglevel error

    # 提取音频为 WAV
    ffmpeg -y -i "$video" -vn -acodec pcm_s16le -ar 16000 -ac 1 "$AUDIO_DIR/${filename}.wav" -loglevel error

    echo "✅ Processed $filename"
done
