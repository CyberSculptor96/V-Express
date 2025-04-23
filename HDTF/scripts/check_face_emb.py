import os
import torch
from tqdm import tqdm

def check_pt_files(directory, output_file):
    invalid_files = []
    pt_files = [f for f in os.listdir(directory) if f.endswith(".pt")]
    
    # 遍历目录下所有 .pt 文件并添加进度条
    for filename in tqdm(pt_files, desc="Checking .pt files"):
        file_path = os.path.join(directory, filename)
        try:
            data = torch.load(file_path)
            if len(data) == 0:
                invalid_files.append(filename)
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            invalid_files.append(filename)  # 发生异常的文件也视为不合法
    
    # 将不合法的文件列表写入输出文件
    with open(output_file, "w") as f:
        for file in invalid_files:
            f.write(file + "\n")
    
    print(f"Check complete. Found {len(invalid_files)} invalid files. Details saved in {output_file}")

if __name__ == "__main__":
    input_directory = "HDTF/new_face_info/"  # 目标目录
    output_filename = "invalid_pt_files.txt"  # 输出文件
    check_pt_files(input_directory, output_filename)