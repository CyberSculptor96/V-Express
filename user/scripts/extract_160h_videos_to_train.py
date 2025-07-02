import os
import json
import shutil
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import random

# 设置路径参数
SRC_DIR = "/shareddisk/yexin/shunian/252_hour_test_480p_49frames/video"
DST_DIR = "/shareddisk/yexin/huanghj/data/TalkVid-160h/videos"
JSON_FILE = "/shareddisk/yexin/shunian/252_hour_test_480p_49frames/videos2caption_cleaned.json"
NUM_FILES = 24000  # 115200
NUM_WORKERS = 64  # 可根据机器性能调整
SHUFFLE = True  # 是否打乱文件顺序

def create_symlink(src_file, dst_dir):
    try:
        filename = os.path.basename(src_file)
        dst_path = os.path.join(dst_dir, filename)
        if not os.path.exists(dst_path):
            os.symlink(src_file, dst_path)
        return True
    except Exception as e:
        return f"Error for {src_file}: {e}"

def main():
    os.makedirs(DST_DIR, exist_ok=True)
    
    with open(JSON_FILE, "r") as f:
        data = json.load(f)
    json_mp4_names = set([item["latent_path"].replace(".pt", ".mp4") for item in data])
    dst_files_set = set(os.listdir(DST_DIR))

    # 获取所有视频文件，按文件名排序，取前 N 个
    all_files = sorted([
        os.path.join(SRC_DIR, f)
        for f in os.listdir(SRC_DIR)
        if f.endswith(".mp4") and f in json_mp4_names and f not in dst_files_set
    ])
    print(f"📂 Found {len(all_files)} video files in {SRC_DIR}.")

    if SHUFFLE:
        print("shuffling files...")
        random.seed(42)
        random.shuffle(all_files)

    selected_files = all_files[:NUM_FILES]

    print(f"🔍 Found {len(selected_files)} files to process. Creating symlinks to {DST_DIR}...")

    # 多进程处理
    with Pool(NUM_WORKERS) as pool:
        with tqdm(total=len(selected_files), desc="Linking") as pbar:
            for _ in pool.imap_unordered(partial(create_symlink, dst_dir=DST_DIR), selected_files):
                pbar.update(1)

    print("✅ Symlink creation completed.")

if __name__ == "__main__":
    main()
