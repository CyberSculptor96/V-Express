import os
import json
from tqdm import tqdm

video_dir = "/shareddisk/yexin/huanghj/data/TalkVid-160h/videos-crop"
face_info_dir = "/shareddisk/yexin/huanghj/data/TalkVid-160h/new_face_info"
audio_embed_dir = "/shareddisk/yexin/huanghj/data/TalkVid-160h/short_clip_aud_embeds"
output_json_path = "data/20h.json"

data = []

for fname in tqdm(os.listdir(face_info_dir), desc="Building JSON"):
    if not fname.endswith(".pt"):
        continue

    base_name = fname.replace(".pt", "")
    video_path = os.path.join(video_dir, base_name + ".mp4")
    face_info_path = os.path.join(face_info_dir, base_name + ".pt")
    audio_embeddings_path = os.path.join(audio_embed_dir, base_name + ".pt")

    # 检查三个文件是否都存在
    if os.path.exists(video_path) and os.path.exists(audio_embeddings_path):
        video_info = {
            "video": video_path,
            "face_info": face_info_path,
            "audio_embeds": audio_embeddings_path
        }
        data.append(video_info)

# 保存为 JSON
with open(output_json_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"\n✅ 完成，共写入 {len(data)} 条记录到 {output_json_path}")
