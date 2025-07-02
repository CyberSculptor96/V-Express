import json

unit = 14400
max_entries = unit * 4
input_json = "data/TalkVid-160h-eng-114255.json"
output_json = f"data/TalkVid-80h-eng-{max_entries}.json"

# 读取原始 JSON 文件
with open(input_json, "r") as f:
    data = json.load(f)

# 提取前 max_entries 条
subset = data[:max_entries]

# 写入新 JSON 文件
with open(output_json, "w") as f:
    json.dump(subset, f, indent=2)

print(f"✅ 提取完成：共写入 {len(subset)} 条数据到 {output_json}")
if len(data) < max_entries:
    print(f"⚠️ 注意：原始 JSON 仅包含 {len(data)} 条记录，未满 {max_entries}")
