import json
import os
from tqdm import tqdm

json_path = "/shareddisk/yexin/huanghj/workspace/V-Express/data/TalkVid-20h-eng-14400.json"
output_txt = "missing_videos.txt"

def main():
    with open(json_path, "r") as f:
        data = json.load(f)

    missing = []
    for item in tqdm(data, desc="Checking video paths"):
        video_path = item.get("video")
        if not os.path.exists(video_path):
            missing.append(video_path)

    if missing:
        with open(output_txt, "w") as f:
            for path in missing:
                f.write(path + "\n")
        print(f"❌ 共找到 {len(missing)} 个缺失文件，已写入 {output_txt}")
    else:
        print("✅ 所有 video 路径文件均存在。")

if __name__ == "__main__":
    main()
