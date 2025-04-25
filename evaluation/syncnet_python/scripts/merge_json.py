import os
import json
from pathlib import Path

# JSON 文件所在目录
json_dir = Path("/wangbenyou/huanghj/workspace/research/V-Express/evaluation/outputs/lip-sync")

# 目标合并后的 JSON 文件
merged_json_path = json_dir / "merged_output.json"

# 存储合并数据的字典
merged_data = {}

# 遍历目录下所有 JSON 文件
for json_file in json_dir.glob("*.json"):
    try:
        with open(json_file, "r") as f:
            data = json.load(f)  # 读取 JSON 文件
            # 处理数据，确保 min_dist 和 confidence 只保留 3 位小数
            for video_key, values in data.items():
                if isinstance(values, dict):
                    values["min_dist"] = round(values.get("min_dist", 0), 3)
                    values["confidence"] = round(values.get("confidence", 0), 3)

                merged_data[video_key] = values  # 更新合并字典
    except json.JSONDecodeError as e:
        print(f"⚠️ 跳过 {json_file}（解析错误: {e}）")

# 将合并后的数据写入新的 JSON 文件
with open(merged_json_path, "w") as f:
    json.dump(merged_data, f, indent=4)

print(f"✅ 合并完成，JSON 文件已保存至: {merged_json_path}")
