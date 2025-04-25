import pickle
import os

# 指定文件路径
file_path = "/wangbenyou/huanghj/workspace/hallo3/evaluation/syncnet_python/output/pywork/1F8VqXsUGjQ-scene21_scene12/activesd.pckl"

# 确保文件存在
if not os.path.exists(file_path):
    print(f"❌ 文件未找到: {file_path}")
else:
    # 读取 pckl 文件
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # 打印数据类型和部分内容
    print(f"✅ 读取成功！数据类型: {type(data)}")
    print(f"📌 数据内容示例: {data}")
