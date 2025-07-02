import argparse
import os, cv2, subprocess
import logging
logging.getLogger("insightface").setLevel(logging.CRITICAL)
os.environ['ORT_LOG_SEVERITY_LEVEL'] = '4'

import shutil
from tqdm import trange
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import time
import numpy as np

from insightface.app import FaceAnalysis

def _get_crop_from_bbox(center, box_size, width, height, size):
    w = int(box_size / 2)
    left = int(center[0] - w)
    top = int(center[1] - w)
    if top < 0:
        top = 0
    if left < 0:
        left = 0
    return [width, height, left, top, 2 * w, 2 * w]

def process_video(vfile, size):
    try:
        basename = os.path.basename(vfile)
        output_file = os.path.join(args.output_dir, basename)
        if os.path.exists(output_file):
            return f"skip {output_file} since it already exists"

        app = FaceAnalysis(
            providers=['CUDAExecutionProvider'],
            provider_options=[{'device_id': 0}],
            allowed_modules=['detection', 'landmark_3d_68']
        )
        app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

        cap = cv2.VideoCapture(vfile)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        ## 由于数据集的视频均为fps=24，无需再进行fps统一处理
        # if fps != 25:
        #     cap.release()
        #     tmp_path =  vfile.replace(".mp4", "_25.mp4")
        #     cmd = f'ffmpeg -y -i "{vfile}" -r 25 "{tmp_path}" -loglevel quiet'
        #     subprocess.call(cmd, shell=True)
        #     os.remove(vfile)
        #     os.rename(tmp_path, vfile)
        #     cap = cv2.VideoCapture(vfile)
        #     fps = 25

        frames = []
        bbox_min = [np.inf, np.inf]
        bbox_max = [-np.inf, -np.inf]

        for _ in range(frame_count):
            still_reading, frame = cap.read()
            if not still_reading:
                break
            preds = app.get(frame)
            if len(preds) == 0:
                continue
            x1, y1, x2, y2 = preds[0].bbox
            bbox_min[0] = min(bbox_min[0], x1)
            bbox_min[1] = min(bbox_min[1], y1)
            bbox_max[0] = max(bbox_max[0], x2)
            bbox_max[1] = max(bbox_max[1], y2)
            frames.append(frame)

        if len(frames) == 0:
            return f"[x] No face detected in {vfile}"

        center_x = int((bbox_min[0] + bbox_max[0]) / 2)
        center_y = int((bbox_min[1] + bbox_max[1]) / 2)
        center = [center_x, center_y]
        box_size = int(max(bbox_max[0] - bbox_min[0], bbox_max[1] - bbox_min[1]) * 1.6)

        cap.release()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_file, fourcc, fps, (size, size))

        for frame in frames:
            crop_info = _get_crop_from_bbox(center, box_size, frame.shape[1], frame.shape[0], size)
            face_img = frame[crop_info[3]:crop_info[3] + crop_info[5], crop_info[2]:crop_info[2] + crop_info[4]]
            face_img = cv2.resize(face_img, (size, size), interpolation=cv2.INTER_NEAREST)
            writer.write(face_img)

        writer.release()
        # ✅ 使用 ffmpeg 转码为 H.264 编码（libx264）
        h264_output_file = output_file.replace(".mp4", "_h264.mp4")
        cmd = f'ffmpeg -y -i "{output_file}" -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p "{h264_output_file}" -loglevel error'
        subprocess.call(cmd, shell=True)
        os.remove(output_file)
        os.rename(h264_output_file, output_file)

        return f"[✓] Saved cropped video to: {output_file}"
    except Exception as e:
        return f"Error processing {vfile}: {e}"

def process_video_star(args):
    return process_video(*args)

def main(args):
    exists_files = set(os.listdir(args.output_dir))

    video_files = sorted([
        os.path.join(args.video_dir, f)
        for f in os.listdir(args.video_dir)
        if f.endswith(".mp4") and f not in exists_files
    ])

    if args.shard:
        process_video_nums = args.videos_per_shard
        curr = args.gpu_id
        video_files = video_files[(curr+8)*process_video_nums:(curr+1+8)*process_video_nums]
        print(f"process video nums: {process_video_nums}")

    tasks = [(vfile, args.size) for vfile in video_files]
    print(f"Found {len(tasks)} video files to process, process with {args.num_workers}.")

    start = time.time()
    with Pool(args.num_workers) as pool:
        with tqdm(total=len(tasks), desc="Processing Videos") as pbar:
            results = []
            for result in pool.imap_unordered(process_video_star, tasks):
                results.append(result)
                pbar.update(1)
    end = time.time()

    for r in results:
        print(r)

    print(f"\n✅ 所有视频处理完成，共耗时 {((end - start) / 60):.2f} 分钟，进程数为 {args.num_workers}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", "-v", type=str, required=True, help="输入视频所在目录")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="输出视频所在目录")
    parser.add_argument("--size", type=int, default=512, help="裁剪后视频的分辨率")
    parser.add_argument("--num_workers", "-w", type=int, default=8, help="并行进程数")
    parser.add_argument("--shard", action="store_false", help="是否对输入的视频文件夹分片")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU ID to use for face detection")
    parser.add_argument("--videos_per_shard", type=int, default=1500, help="")
    args = parser.parse_args()

    main(args)
