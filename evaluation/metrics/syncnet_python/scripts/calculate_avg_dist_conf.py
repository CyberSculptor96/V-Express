import json
import os.path as osp
from pathlib import Path

# 指定 syncnet.json 文件路径
name = "gt-176"
base_path = "/wangbenyou/huanghj/workspace/research/V-Express"
path = osp.join(base_path, "evaluation/outputs/lip-sync/json")
json_path = osp.join(path, f"{name}.json")

# 指定输出 TXT 文件路径
output_txt_path = osp.join(path, f"{name}.txt")

# 确保 syncnet.json 存在
if not Path(json_path).exists():
    print(f"❌ 文件未找到: {json_path}")
    exit()

# 读取 JSON 文件
with open(json_path, "r") as f:
    data = json.load(f)

# 初始化变量
total_min_dist = 0
total_confidence = 0
count = 0

# 初始化 min/max 变量
min_min_dist = float("inf")
max_min_dist = float("-inf")
min_confidence = float("inf")
max_confidence = float("-inf")

# 遍历 JSON 数据，计算总和、最小值和最大值
for video_key, values in data.items():
    if isinstance(values, dict):
        min_dist = values.get("min_dist", 0)
        confidence = values.get("confidence", 0)

        total_min_dist += min_dist
        total_confidence += confidence
        count += 1

        # 更新最小值和最大值
        min_min_dist = min(min_min_dist, min_dist)
        max_min_dist = max(max_min_dist, min_dist)
        min_confidence = min(min_confidence, confidence)
        max_confidence = max(max_confidence, confidence)

# 计算平均值
if count > 0:
    avg_min_dist = round(total_min_dist / count, 3)
    avg_confidence = round(total_confidence / count, 3)
else:
    avg_min_dist = 0
    avg_confidence = 0

# 打印最小值、最大值和平均值
print(f"📊 统计结果：")
print(f"➡️  min_dist: 平均值 = {avg_min_dist}, 最小值 = {min_min_dist}, 最大值 = {max_min_dist}")
print(f"➡️  confidence: 平均值 = {avg_confidence}, 最小值 = {min_confidence}, 最大值 = {max_confidence}")

# 写入 TXT 文件
with open(output_txt_path, "w") as f:
    f.write(f"Average min_dist: {avg_min_dist}\n")
    f.write(f"Min min_dist: {min_min_dist}\n")
    f.write(f"Max min_dist: {max_min_dist}\n")
    f.write(f"Average confidence: {avg_confidence}\n")
    f.write(f"Min confidence: {min_confidence}\n")
    f.write(f"Max confidence: {max_confidence}\n")

print(f"✅ 计算完成，平均值、最小值和最大值已写入: {output_txt_path}")
