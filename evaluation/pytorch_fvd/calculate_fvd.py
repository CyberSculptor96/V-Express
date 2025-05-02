import numpy as np
import torch
from tqdm import tqdm
import os
import cv2
import multiprocessing
from functools import partial
from pathlib import Path

def trans(x):
    # if greyscale images add channel
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)

    # permute BTCHW -> BCTHW
    x = x.permute(0, 2, 1, 3, 4) 

    return x

def calculate_fvd(videos1, videos2, device, method='styleganv', only_final=False):

    if method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    print("calculate_fvd...")

    # videos [batch_size, timestamps, channel, h, w]
    
    min_batch_size = min(videos1.shape[0], videos2.shape[0])
    min_timestamps = min(videos1.shape[1], videos2.shape[1])

    # 截取最小的 batch 和 timestamps，确保维度一致
    videos1 = videos1[:min_batch_size, :min_timestamps]
    videos2 = videos2[:min_batch_size, :min_timestamps]

    print(f"Adjusted videos1.shape: {videos1.shape}")
    print(f"Adjusted videos2.shape: {videos2.shape}")

    assert videos1.shape == videos2.shape

    i3d = load_i3d_pretrained(device=device)
    fvd_results = []

    # support grayscale input, if grayscale -> channel*3
    # BTCHW -> BCTHW
    # videos -> [batch_size, channel, timestamps, h, w]

    videos1 = trans(videos1)
    videos2 = trans(videos2)

    fvd_results = []

    if only_final:

        assert videos1.shape[2] >= 10, "for calculate FVD, each clip_timestamp must >= 10"

        # videos_clip [batch_size, channel, timestamps, h, w]
        videos_clip1 = videos1
        videos_clip2 = videos2

        # get FVD features
        feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
        feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)

        # calculate FVD
        fvd_results.append(frechet_distance(feats1, feats2))
    
    else:

        # for calculate FVD, each clip_timestamp must >= 10
        for clip_timestamp in tqdm(range(10, videos1.shape[-3]+1)):
        
            # get a video clip
            # videos_clip [batch_size, channel, timestamps[:clip], h, w]
            videos_clip1 = videos1[:, :, : clip_timestamp]
            videos_clip2 = videos2[:, :, : clip_timestamp]

            # get FVD features
            feats1 = get_fvd_feats(videos_clip1, i3d=i3d, device=device)
            feats2 = get_fvd_feats(videos_clip2, i3d=i3d, device=device)
        
            # calculate FVD when timestamps[:clip]
            fvd_results.append(frechet_distance(feats1, feats2))

    result = {
        "value": fvd_results,
    }

    return result


def main():
    # 真实视频路径和生成视频路径
    real_video_path = "/wangbenyou/huanghj/workspace/research/V-Express/output/evaluations/gt-50"
    # gen_video_path = "/wangbenyou/huanghj/workspace/hallo3/output/output_testset"
    gen_video_path = "/wangbenyou/huanghj/workspace/research/V-Express/output/evaluations/wav2lip-50"

    nums_of_video_real = sum(1 for root, _, files in os.walk(real_video_path) for f in files if f.endswith(".mp4"))
    nums_of_video_gen = sum(1 for root, _, files in os.walk(gen_video_path) for f in files if f.endswith(".mp4"))
    print(f"{nums_of_video_real=}, {nums_of_video_gen=}")
    NUMBER_OF_VIDEOS = min(nums_of_video_real, nums_of_video_gen, 200)

    # video_length_real = 50
    # video_length_gen = 125
    # VIDEO_LENGTH = min(video_length_real, video_length_gen)
    VIDEO_LENGTH = 101

    # NUMBER_OF_VIDEOS = 8
    CHANNEL = 3
    # SIZE = 64
    SIZE = 360

    # 加载真实视频和生成视频
    real_videos = load_video_from_path(real_video_path, NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, SIZE)
    generated_videos = load_video_from_path(gen_video_path, NUMBER_OF_VIDEOS,  VIDEO_LENGTH, CHANNEL, SIZE)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 计算 FVD 分数
    result = calculate_fvd(real_videos, generated_videos, device, method='videogpt', only_final=True)
    print("[fvd-videogpt ]", result["value"])

    result = calculate_fvd(real_videos, generated_videos, device, method='styleganv', only_final=True)
    print("[fvd-styleganv]", result["value"])


def get_video_files(video_path, num_videos):
    """
    获取所有 .mp4 文件路径，支持二级目录，并返回前 num_videos 个文件
    """
    video_files = []

    # 遍历一级目录
    for entry in os.scandir(video_path):
        if entry.is_file() and entry.name.endswith(".mp4"):
            video_files.append(entry.path)
        elif entry.is_dir():
            sub_files = [
                os.path.join(entry.path, f) for f in os.listdir(entry.path) if f.endswith(".mp4")
            ]
            video_files.extend(sub_files)

    # 按字母顺序排序，确保一致性
    if os.path.basename(video_path) == "videos":
        video_files = sorted(video_files)[:num_videos]
    elif os.path.basename(video_path) == "media":
        video_files = sorted(video_files)[-num_videos:]
    else:
        video_files = sorted(video_files)[:num_videos]
    
    if len(video_files) == 0:
        raise ValueError(f"目录 {video_path} 下没有找到 .mp4 视频文件！")
    
    print(f"Found {len(video_files)} video files.")
    return video_files

def process_video(video_file, video_length, size):
    """
    读取单个视频并转换为张量格式
    """
    if not os.path.exists(video_file):
        print(f"视频文件 {video_file} 不存在，跳过。")
        return None

    cap = cv2.VideoCapture(video_file)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < video_length:
        # print(f"视频 {video_file} 的帧数不足（{total_frames} < {video_length}），跳过。")
        cap.release()
        return None

    frames = []
    for _ in range(video_length):
        ret, frame = cap.read()
        if not ret:
            print(f"视频 {video_file} 读取失败，跳过。")
            cap.release()
            return None
        frame = cv2.resize(frame, (size, size))  # 调整帧大小
        frame = frame.transpose(2, 0, 1)  # (H, W, C) → (C, H, W)
        frames.append(frame)

    cap.release()
    return np.array(frames)  # 返回 numpy 数组

def load_video_from_path(video_path, num_videos, video_length, channel, size, num_workers=64):
    """
    并行加载视频并转换为张量格式
    """
    # 获取视频文件列表
    video_files = get_video_files(video_path, num_videos)

    results = []

    # # 进度条 & 多进程
    with multiprocessing.Pool(num_workers) as pool:
        process_with_args = partial(process_video, video_length=video_length, size=size)  # 预定义参数
        results = list(tqdm(pool.imap(process_with_args, video_files),
                            total=len(video_files), unit="video"))

    # # 单进程 for 循环版本，保留 tqdm 进度条
    # for video_file in tqdm(video_files, total=len(video_files), unit="video"):
    #     result = process_video(video_file, video_length=video_length, size=size)
    #     results.append(result)

    # 过滤 None 值
    videos = [res for res in results if res is not None]

    if len(videos) == 0:
        raise ValueError("所有视频处理失败！")

    videos = np.array(videos)  # 转换为 numpy 数组
    videos = torch.tensor(videos, dtype=torch.float32) / 255.0
    print(f"{torch.min(videos)=}, {torch.max(videos)=}")

    # torch.save(videos, f"{Path(video_path).stem}.pth")
    print(f"{videos.shape=}")
    return videos

if __name__ == "__main__":
    main()

# def load_video_from_path(video_path, num_videos, video_length, channel, size):
#     """
#     从指定路径加载视频并转换为张量格式
#     :param video_path: 视频文件路径
#     :param num_videos: 视频数量
#     :param video_length: 视频长度 (帧数）
#     :param channel: 视频通道数 (例如 3 表示 RGB)
#     :param size: 视频帧的尺寸 (例如 64x64)
#     :return: 视频张量 (num_videos, video_length, channel, size, size)
#     """
#     # 获取所有 .mp4 文件，并按文件名排序（确保顺序一致）
#     video_files = []

#     # 遍历一级目录
#     for entry in os.scandir(video_path):
#         if entry.is_file() and entry.name.endswith(".mp4"):
#             # 直接在 video_path 目录下的 .mp4 文件
#             video_files.append(entry.path)
#         elif entry.is_dir():
#             # 在子目录下查找 .mp4 文件
#             sub_files = [
#                 os.path.join(entry.path, f) for f in os.listdir(entry.path) if f.endswith(".mp4")
#             ]
#             video_files.extend(sub_files)

#     video_files = sorted(video_files)
#     video_files = video_files[:num_videos]
#     print(f"{len(video_files)=}")

#     if len(video_files) == 0:
#         raise ValueError(f"目录 {video_path} 下没有找到 .mp4 视频文件！")
    
#     videos = []
#     for video_file in video_files:
#         if not os.path.exists(video_file):
#             print(f"视频文件 {video_file} 不存在，跳过。")
#             continue

#         cap = cv2.VideoCapture(video_file)
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数

#         if total_frames < video_length:
#             print(f"视频 {video_file} 的帧数不足（{total_frames} < {video_length}），跳过。")
#             cap.release()
#             continue

#         frames = []
#         for _ in range(video_length):
#             ret, frame = cap.read()
#             if not ret:
#                 raise ValueError(f"视频 {video_file} 的帧数不足 {video_length}")
#             frame = cv2.resize(frame, (size, size))  # 调整帧大小
#             frame = frame.transpose(2, 0, 1)  # 将 (H, W, C) 转换为 (C, H, W)
#             frames.append(frame)
#         cap.release()
#         videos.append(frames)
#     videos = np.array(videos)  # 转换为 numpy 数组
#     videos = torch.tensor(videos, dtype=torch.float32)  # 转换为 PyTorch 张量
#     print(f"{videos.shape=}")
#     return videos