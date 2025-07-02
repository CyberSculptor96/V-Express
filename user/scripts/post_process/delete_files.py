import os
from tqdm import tqdm

# 输入路径
video_dir = "/shareddisk/yexin/huanghj/data/TalkVid-160h/videos"
audio_emb_dir = "/shareddisk/yexin/huanghj/data/TalkVid-160h/new_face_info"
txt_path = "/shareddisk/yexin/huanghj/workspace/V-Express/lang_zh.txt"

# 读取文件名列表（确保剔除空行与空格）
with open(txt_path, "r") as f:
    filenames = [line.strip() for line in f if line.strip()]

deleted = []
not_found = []

for name in tqdm(filenames):
    video_path = os.path.join(audio_emb_dir, name.replace(".mp4", ".pt"))
    if os.path.exists(video_path):
        os.remove(video_path)
        deleted.append(name)
    else:
        not_found.append(name)

# 打印总结
print(f"✅ 删除完成：{len(deleted)} 个文件")
print(f"⚠️ 未找到文件：{len(not_found)} 个")

if not_found:
    print("📋 以下文件未找到：")
    for name in not_found[:10]:  # 只打印前10个，避免刷屏
        print(f" - {name}")

# txt_file = "/shareddisk/yexin/huanghj/workspace/V-Express/160h_frames_mismatch.txt"

# with open(txt_file, "r") as f:
#     lines = f.readlines()

# for line in lines:
#     path = line.split(" | ")[0].strip()
#     if path and path.endswith(".mp4"):
#         try:
#             os.remove(path)
#             print(f"✅ Deleted: {path}")
#         except Exception as e:
#             print(f"❌ Failed to delete {path}: {e}")

