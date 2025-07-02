#!/bin/bash
gpu_id=1
export CUDA_VISIBLE_DEVICES=$gpu_id

# ==== 参数设置 ====
VIDEO_DIR="data/test"           # 输入视频目录（只包含.mp4）
OUTPUT_DIR="data/crop"         # 输出视频目录（将自动创建）
SIZE=512                                     # 裁剪后的视频分辨率（正方形）
NUM_WORKERS=16                                # 并行进程数
NUM_SHARDS=16

# ==== 创建输出目录 ====
mkdir -p "$OUTPUT_DIR"

# ==== 启动 Python 脚本 ====
python user/scripts/face_crop.py \
    --video_dir "$VIDEO_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --size "$SIZE" \
    --num_workers "$NUM_WORKERS"

#!/bin/bash

# 通用参数
VIDEO_DIR="/your/full/video/dir"
OUTPUT_DIR="/your/output/dir"
NUM_SHARDS=16

# 所有视频路径列表
ALL_VIDEOS="video_list.txt"
find "$VIDEO_DIR" -name "*.mp4" | sort > $ALL_VIDEOS

LINES=$(wc -l < $ALL_VIDEOS)
PER_SHARD=$(( ($LINES + $NUM_SHARDS - 1) / $NUM_SHARDS ))

for (( i=0; i<$NUM_SHARDS; i++ ))
do
    START=$(( $i * $PER_SHARD + 1 ))
    END=$(( ($i + 1) * $PER_SHARD ))

    # 生成每个 shard 的路径列表
    SHARD_LIST="video_list_$i.txt"
    sed -n "${START},${END}p" $ALL_VIDEOS > $SHARD_LIST

    # 启动子任务
    CUDA_VISIBLE_DEVICES=$i \
    python user/scripts/face_crop.py \
        --video_list $SHARD_LIST \
        --output_dir "$OUTPUT_DIR" \
        --size "$SIZE" \
        --num_workers "$NUM_WORKERS" &

done

wait
echo "✅ All shards finished."

