import os
import torch
import cv2
from insightface.app import FaceAnalysis

# 模型文件目录
model_root_path = './model_ckpts/insightface_models/'

# 输入视频所在目录
input_dir = '/wangbenyou/huanghj/workspace/research/V-Express/HDTF/short_clip'
# 输出人脸信息文件所在目录
output_dir = '/wangbenyou/huanghj/workspace/research/V-Express/HDTF/new_face_info'
os.makedirs(output_dir, exist_ok=True)

# 初始化人脸分析模型
app = FaceAnalysis(
    providers=['CUDAExecutionProvider'],
    provider_options=[{'device_id': 0}],
    root=model_root_path,
)
app.prepare(ctx_id=0, det_size=(512, 512))

# 读取无效的 .pt 文件列表
invalid_files_path = "invalid_pt_files.txt"
with open(invalid_files_path, "r") as f:
    invalid_files = {line.strip() for line in f.readlines()}

# 定义起始文件
start_file = 'WRA_TomPrice_000_clip_9.mp4'
start_processing = True

# 获取并按字典序（sorted）列出所有 .mp4 文件
video_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mp4')])
video_files = sorted([f.replace('.pt', '.mp4') for f in invalid_files if f.endswith('.pt')])
print(f'Found {len(video_files)} video files')

# 遍历所有视频文件并处理
for video_file in video_files:
    # 检查是否达到了起始文件
    if video_file == start_file:
        start_processing = True

    # 如果还没到起始文件，继续下一个视频
    if not start_processing:
        continue
    
    # 构建视频路径
    vid_path = os.path.join(input_dir, video_file)
    # 构建输出 .pt 文件路径（将 .mp4 后缀改为 .pt）
    face_info_path = os.path.join(output_dir, video_file.replace('.mp4', '.pt'))

    print(f'Processing video: {vid_path}')

    # 读取所有帧
    frames = []
    video_capture = cv2.VideoCapture(vid_path)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frames.append(frame)
    video_capture.release()

    # 逐帧进行人脸检测
    face_info = []
    drop_flag = False
    for frame in frames:
        faces = app.get(frame)
        # 如果一帧检测到的人脸数目不是1，则标记为 drop_flag
        if len(faces) != 1:
            drop_flag = True
            break

        if not drop_flag:
            # 将检测到的人脸信息存入 face_info
            face_info.append([
                {
                    'bbox': face.bbox,
                    'kps': face.kps,
                    'det_score': face.det_score,
                    'landmark_3d_68': face.landmark_3d_68,
                    'pose': face.pose,
                    'landmark_2d_106': face.landmark_2d_106,
                    'gender': face.gender,
                    'age': face.age,
                    'embedding': face.embedding,
                }
                for face in faces
            ])
        else:
            print('error', vid_path)

    # 将 face_info 保存为 .pt 文件
    torch.save(face_info, face_info_path)
    print(f'Saving face info to {face_info_path}')
