import os

# 读取无效的 .pt 文件列表
invalid_files_path = "invalid_pt_files.txt"
with open(invalid_files_path, "r") as f:
    invalid_files = [line.strip() for line in f.readlines()]

# 输出人脸信息文件所在目录
output_dir = '/wangbenyou/huanghj/workspace/research/V-Express/HDTF/new_face_info'

# 删除无效的 .pt 文件
for pt_file in invalid_files:
    file_path = os.path.join(output_dir, pt_file)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"File not found: {file_path}")

print("Deletion process completed.")
