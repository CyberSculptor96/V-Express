import os
import cv2
import multiprocessing as mp
from glob import glob
from tqdm import tqdm

# ✅ 设置参数
VIDEO_DIR = "/shareddisk/yexin/huanghj/data/TalkVid-160h/videos-crop"
TARGET_FPS = 24.0
FRAME_COUNTS = 121
NUM_WORKERS = 64   # 根据 CPU 核数调整

# ✅ 每个进程处理的视频检查函数
def check_fps(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return (video_path, None, "Failed to open")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if abs(fps - TARGET_FPS) > 0.1 or frames != FRAME_COUNTS:
            return (video_path, fps, frames, "Mismatch")
        return None
    except Exception as e:
        return (video_path, None, None, str(e))

# ✅ 主逻辑
def main():
    video_paths = glob(os.path.join(VIDEO_DIR, "*.mp4"))
    print(f"Found {len(video_paths)} videos in {VIDEO_DIR}.")

    with mp.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(check_fps, video_paths), total=len(video_paths)))

    mismatched = [r for r in results if r is not None]
    print(f"\n🟡 Found {len(mismatched)} videos with fps mismatch:")

    for path, fps, frames, reason in mismatched:
        print(f" - {path} | Detected FPS: {fps} | Frames: {frames} | Reason: {reason}")

if __name__ == "__main__":
    main()
