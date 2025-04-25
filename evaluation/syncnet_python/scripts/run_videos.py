import os
import subprocess
import multiprocessing
from tqdm import tqdm

# 输入路径
TESTSET_DIR = "/wangbenyou/huanghj/workspace/research/V-Express/output/evaluations/gt-176"
DATA_DIR = "/wangbenyou/huanghj/workspace/research/V-Express/evaluation/syncnet_python/output"
process_dir = False

# 运行单个视频的任务
def process_video(video):
    if process_dir:
        video_path = os.path.join(TESTSET_DIR, video, f"000000_with_audio.mp4")
    else:
        video_path = os.path.join(TESTSET_DIR, video)

    # 确保视频文件存在
    if not os.path.exists(video_path):
        print(f"❌ 视频文件未找到: {video_path}")
        return
    
    print(f"🚀 处理视频: {video_path}")

    # 构造命令
    pipeline_cmd = [
        "python", "run_pipeline.py",
        "--videofile", video_path,
        "--reference", os.path.splitext(video)[0],
        "--data_dir", DATA_DIR
    ]
    
    syncnet_cmd = [
        "python", "run_syncnet.py",
        "--videofile", video_path,
        "--reference", os.path.splitext(video)[0],
        "--data_dir", DATA_DIR
    ]

    # 执行 `run_pipeline.py`
    print(f"▶️ 运行: {' '.join(pipeline_cmd)}")
    subprocess.run(pipeline_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 执行 `run_syncnet.py`
    print(f"▶️ 运行: {' '.join(syncnet_cmd)}")
    subprocess.run(syncnet_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    print(f"✅ 处理完成: {video_path}")

if process_dir:
    videos = [d for d in os.listdir(TESTSET_DIR) if os.path.isdir(os.path.join(TESTSET_DIR, d))]
else:
    videos = [f for f in sorted(os.listdir(TESTSET_DIR)) if f.endswith(".mp4") and os.path.isfile(os.path.join(TESTSET_DIR, f))]
# 设置多进程池
NUM_PROCESSES = 32

if __name__ == "__main__":
    # with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
    #     pool.map(process_video, video_folders)
    with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
        # 使用 `tqdm` 进行进度可视化
        for result in tqdm(pool.imap_unordered(process_video, videos), total=len(videos), desc="进度"):
            print(result)

    print("🎉 所有视频处理完成！")
