import os
import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 输入路径
video_dir = "/shareddisk/yexin/huanghj/data/TalkVid-160h/videos-crop"
json_path = "/shareddisk/yexin/shunian/252_hour_test_480p_49frames/videos2caption_cleaned.json"
output_txt = "missing_videos.txt"

# 解析 JSON 中已有的 .pt 文件名（转为 .mp4 后缀）
with open(json_path, "r") as f:
    data = json.load(f)
json_mp4_names = set([item["latent_path"].replace(".pt", ".mp4") for item in data])

# 视频目录下的所有 .mp4 文件名（不含路径）
all_videos = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])

# 多进程检查是否不在 JSON 列表中
def check_missing(video_name):
    if video_name not in json_mp4_names:
        return video_name
    return None

if __name__ == "__main__":
    with Pool(64) as pool:
        results = list(tqdm(pool.imap_unordered(check_missing, all_videos), total=len(all_videos)))

    missing_videos = [r for r in results if r is not None]

    # 写入输出文件
    with open(output_txt, "w") as f:
        for name in missing_videos:
            f.write(name + "\n")

    print(f"✅ Missing video count: {len(missing_videos)}")
    print(f"📝 Output written to: {output_txt}")
