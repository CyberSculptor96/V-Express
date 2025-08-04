#!/bin/bash
# ==== 可配置参数 ====
ROOT="/wangbenyou/huanghj/workspace/V-Express"    # 项目根目录
cd "$ROOT"

TEST_DATASET="hallo3"                   # 测试集名称, e.g., TalkVid, HDTF
TRAINING_DATASET="TalkVid-160h"         # 训练集名称, e.g., TalkVid, HDTF
DATA_PREFIX="$ROOT/evaluation/testset"  # 存放测试集的路径前缀
INPUT_IMG_DIR="$DATA_PREFIX/$TEST_DATASET/data/imgs"          # 第一帧图像文件夹
INPUT_AUDIO_DIR="$DATA_PREFIX/$TEST_DATASET/data/audios"      # 驱动音频文件夹
INPUT_VKPS_DIR="$DATA_PREFIX/$TEST_DATASET/data/vkps"         # 驱动关键点文件夹
OUTPUT_DIR="$DATA_PREFIX/$TEST_DATASET/trainset/$TRAINING_DATASET/videos"   # 输出视频文件夹
LOG_DIR="$ROOT/logs/inference"              # 日志目录
MODEL_DIR="$ROOT/exp/$TRAINING_DATASET"     # 推理的模型权重存放目录
MODEL_ROOT="$ROOT/model_ckpts/insightface_models"
NUM_GPUS=8                  # 使用的GPU数量
USE_TRAINING_CKPTS="true"   # 是否使用训练集的模型权重，还是使用V-Express的模型权重
STEPS=50000                 # 训练集模型的训练步数
echo -e "数据集: $TEST_DATASET, \n输入图像目录: $INPUT_IMG_DIR, \n输入音频目录: $INPUT_AUDIO_DIR, \n输出目录: $OUTPUT_DIR"

# ==== 设置可见的GPU设备 ====
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "$(dirname "$0")"  # 切换到脚本所在目录
mkdir -p "$OUTPUT_DIR"  # 创建输出目录
mkdir -p "$LOG_DIR"  # 创建日志目录
mkdir -p "$MODEL_ROOT"  # 确保模型目录存在

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "创建输出目录失败: $OUTPUT_DIR"
    exit 1
fi

# 获取所有文件，并排序
all_ref_imgs=($(ls ${INPUT_IMG_DIR}/*.jpg | sort))
all_audios=($(ls ${INPUT_AUDIO_DIR}/*.wav | sort))
all_vkps=($(ls ${INPUT_VKPS_DIR}/*.pt | sort))

all_ref_imgs=("${all_ref_imgs[@]:0:100}")
all_audios=("${all_audios[@]:0:100}")
all_vkps=("${all_vkps[@]:0:100}")
# all_files_num=($(ls ${INPUT_IMG_DIR}/*.jpg | wc -l))

# 获取已生成的 MP4 文件列表
existing_videos=($(ls ${OUTPUT_DIR}/*.mp4))

# 将现有的 .mp4 文件名存入数组
existing_video_bases=()
for video in "${existing_videos[@]}"; do
    existing_video_bases+=("$(basename "$video" .mp4)")
done

# 初始化一个空数组，用来存储待处理的文件集合
to_process_ref_imgs=()
to_process_audios=()
to_process_vkps=()

# 遍历图像、音频和关键点文件，筛选出尚未生成对应视频的文件
for i in "${!all_ref_imgs[@]}"; do
    ref_img="${all_ref_imgs[$i]}"
    audio="${all_audios[$i]}"
    vkps="${all_vkps[$i]}"

    # 检查对应的 .mp4 文件是否已存在
    base_name=$(basename "$ref_img" .jpg)
    if [[ " ${existing_video_bases[@]} " =~ " ${base_name} " ]]; then
        echo "文件 ${base_name}.mp4 已存在，跳过生成"
        continue
    fi

    # 如果文件尚未存在，加入待处理数组
    to_process_ref_imgs+=("$ref_img")
    to_process_audios+=("$audio")
    to_process_vkps+=("$vkps")
done

# ==== 启动每张卡的任务 ====
for (( i=0; i<$NUM_GPUS; i++ ))
do
    gpu_id=$i
    # 计算当前 GPU 需要处理的索引范围
    total_samples=${#to_process_ref_imgs[@]}
    samples_per_gpu=$((total_samples / NUM_GPUS))
    start_index=$((gpu_id * samples_per_gpu))
    end_index=$((start_index + samples_per_gpu - 1))

    # 确保索引不超出范围
    if [[ $start_index -ge $total_samples ]]; then
        echo "错误：GPU ${gpu_id} 无可用样本，跳过推理"
        exit 0
    fi
    if [[ $end_index -ge $total_samples ]]; then
        end_index=$((total_samples - 1))
    fi

    # 选取当前 GPU 负责的样本（确保不超出范围）
    num_samples=$((end_index - start_index + 1))
    ref_imgs=("${to_process_ref_imgs[@]:$start_index:$num_samples}")
    audios=("${to_process_audios[@]:$start_index:$num_samples}")
    vkps=("${to_process_vkps[@]:$start_index:$num_samples}")
    log_path="$LOG_DIR/inference_${gpu_id}.log"

    echo "==============================="
    echo "开始推理任务"
    echo "GPU $gpu_id → 视频数量: $num_samples → 日志: $log_path"
    echo "处理样本：    ${start_index} to ${end_index}"
    echo "总样本数:     ${#ref_imgs[@]} Images, ${#audios[@]} Audios"
    echo "==============================="

    echo "Using model_ckpts"
    if [[ "$USE_TRAINING_CKPTS" == "true" ]]; then
        echo "Using training model checkpoints"
        denoising_unet_path="$MODEL_DIR/stage_3/denoising_unet-$STEPS.pth"
        reference_net_path="$MODEL_DIR/stage_3/reference_net-$STEPS.pth"
        v_kps_guider_path="$MODEL_DIR/stage_3/v_kps_guider-$STEPS.pth"
        audio_projection_path="$MODEL_DIR/stage_3/audio_projection-$STEPS.pth"
        motion_module_path="$MODEL_DIR/stage_3/motion_module-$STEPS.pth"
    else
        echo "Using V-Express model checkpoints"
        denoising_unet_path="./model_ckpts/v-express/denoising_unet.bin"
        reference_net_path="./model_ckpts/v-express/reference_net.bin"
        v_kps_guider_path="./model_ckpts/v-express/v_kps_guider.bin"
        audio_projection_path="./model_ckpts/v-express/audio_projection.bin"
        motion_module_path="./model_ckpts/v-express/motion_module.bin"
    fi
    
    # 启动后台进程进行推理任务
    export PYTHONWARNINGS="ignore::FutureWarning"
    (for i in "${!ref_imgs[@]}"; do
        reference_image_path="${ref_imgs[$i]}"
        audio_path="${audios[$i]}"
        vkps_path="${vkps[$i]}"

        base_name=$(basename "$reference_image_path" .jpg)
        output_path="$OUTPUT_DIR/${base_name}.mp4"

        echo "-------------------------------"
        echo "Run $((i+1)) on GPU ${gpu_id}"
        echo "Reference Image: ${reference_image_path}"
        echo "Audio: ${audio_path}"
        echo "Vkps: ${vkps_path}"
        echo "Output: ${output_path}"
        echo "Log: ${log_path}"
        echo "-------------------------------"

        # 遍历所有路径，逐个判断文件是否存在
        required_files=("$reference_image_path" "$audio_path" "$vkps_path" "$log_path")
        for f in "${required_files[@]}"; do
            if [[ ! -f "$f" ]]; then
                echo "缺失文件: $f，终止 GPU ${gpu_id} 当前样本的推理"
                continue
            fi
        done

        CUDA_VISIBLE_DEVICES=$gpu_id python inference.py \
            --reference_image_path "$reference_image_path" \
            --audio_path "$audio_path" \
            --kps_path "$vkps_path" \
            --output_path "$output_path" \
            --denoising_unet_path "$denoising_unet_path" \
            --reference_net_path "$reference_net_path" \
            --v_kps_guider_path "$v_kps_guider_path" \
            --audio_projection_path "$audio_projection_path" \
            --motion_module_path "$motion_module_path" \
            --retarget_strategy "naive_retarget" \
            --num_inference_steps 25 \
            --guidance_scale 3.5 \
            --audio_attention_weight 2.0 \
            --context_frames 24 \
            --test_stage "stage_3" \
            > "$log_path" 2>&1

        echo "Run $((i+1)) Completed ✅"
    done

    echo "==============================="
    echo "All ${#ref_imgs[@]} Inference Runs Completed on GPU ${gpu_id}"
    echo "==============================="
    ) &  # 后台执行每个GPU任务
done

# 等待所有后台进程完成
wait

echo "==============================="
echo "所有推理任务完成"
echo "==============================="