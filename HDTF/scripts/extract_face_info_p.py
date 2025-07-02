import os
import cv2
import torch
from tqdm import tqdm
from insightface.app import FaceAnalysis
from multiprocessing import Pool
import argparse

DET_SIZE = 640  # 512
def process_video(args):
    video_file, input_dir, output_dir, model_root_path, gpu_id = args

    try:
        # 设置当前进程使用的 CUDA 设备
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        app = FaceAnalysis(
            providers=['CUDAExecutionProvider'],
            provider_options=[{'device_id': 0}],
            root=model_root_path,
        )
        app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))

        vid_path = os.path.join(input_dir, video_file)
        face_info_path = os.path.join(output_dir, video_file.replace('.mp4', '.pt'))
        if os.path.exists(face_info_path):
            return f"[✓] Skipped: {video_file}"

        # 读取帧
        frames = []
        cap = cv2.VideoCapture(vid_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        face_info = []
        for frame in frames:
            faces = app.get(frame)
            if len(faces) != 1:
                return f"[x] Dropped: {video_file} (face count != 1)"
            face_info.append([{
                'bbox': face.bbox,
                'kps': face.kps,
                'det_score': face.det_score,
                'landmark_3d_68': face.landmark_3d_68,
                'pose': face.pose,
                'landmark_2d_106': face.landmark_2d_106,
                'gender': face.gender,
                'age': face.age,
                'embedding': face.embedding,
            } for face in faces])

        torch.save(face_info, face_info_path)
        return f"[✓] Saved: {video_file}"
    except Exception as e:
        return f"[x] Failed: {video_file} | {e}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_root', type=str, default='./model_ckpts/insightface_models/')
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--videos_per_shard', type=int, default=7200)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    ## 二次处理逻辑
    face_base = "/shareddisk/yexin/huanghj/data/TalkVid-160h/new_face_info"
    face_file = set(os.listdir(face_base))
    video_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.mp4') and f.replace(".mp4", ".pt") not in face_file])
    print(f"Found {len(video_files)} videos to process in {args.input_dir}.")

    start = (args.gpu_id) * args.videos_per_shard
    end = (args.gpu_id + 1) * args.videos_per_shard
    shard_files = video_files[start:end]

    print(f"GPU {args.gpu_id} | Processing {len(shard_files)} videos...")

    task_args = [(f, args.input_dir, args.output_dir, args.model_root, args.gpu_id) for f in shard_files]
    with Pool(args.num_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_video, task_args), total=len(task_args)):
            print(result)

if __name__ == "__main__":
    main()
