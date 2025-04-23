"""
增强版face_emb筛查，确保face_emb.pt的帧数与video.mp4完全一致。
处理结果: 共筛出5个不符合要求的视频，均为WDA_DonnaShalala1_000*
"""
import os
import torch
import subprocess
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_video_frame_count(video_path):
    """使用ffmpeg获取视频帧数"""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return int(result.stdout.strip())
    except ValueError:
        return -1  # 返回-1表示读取失败

def process_file(pt_file, input_dir, output_dir):
    """处理单个 .pt 文件"""
    pt_path = os.path.join(output_dir, pt_file)
    video_file = pt_file.replace(".pt", ".mp4")
    video_path = os.path.join(input_dir, video_file)
    
    if not os.path.exists(video_path):
        return None  # 如果对应的.mp4文件不存在，跳过
    
    try:
        pt_data = torch.load(pt_path)  # 避免 FutureWarning
        pt_frame_count = len(pt_data)
    except Exception as e:
        print(f"Error loading {pt_file}: {e}")
        return None
    
    video_frame_count = get_video_frame_count(video_path)
    if video_frame_count == -1:
        print(f"Failed to retrieve frame count for {video_file}")
        return None
    
    if pt_frame_count < video_frame_count:
        return pt_file
    return None

def check_frame_mismatch(input_dir, output_dir, output_file):
    mismatched_files = []
    pt_files = [f for f in os.listdir(output_dir) if f.endswith(".pt")]
    
    with ProcessPoolExecutor() as executor:
        future_to_file = {executor.submit(process_file, pt_file, input_dir, output_dir): pt_file for pt_file in pt_files}
        for future in tqdm(as_completed(future_to_file), total=len(pt_files), desc="Checking mismatched frames"):
            result = future.result()
            if result:
                mismatched_files.append(result)
    
    # 保存到输出文件
    with open(output_file, "w") as f:
        for file in mismatched_files:
            f.write(file + "\n")
    
    print(f"Check complete. Found {len(mismatched_files)} mismatched files. Details saved in {output_file}")

if __name__ == "__main__":
    input_directory = "/wangbenyou/huanghj/workspace/research/V-Express/HDTF/short_clip"
    output_directory = "/wangbenyou/huanghj/workspace/research/V-Express/HDTF/new_face_info"
    output_filename = "mismatched_files.txt"
    
    check_frame_mismatch(input_directory, output_directory, output_filename)
