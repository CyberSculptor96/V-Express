import torch
import torchvision
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import os

# 准备音频嵌入的函数
def prepare_audio_embeddings(audio_waveform, audio_processor, audio_encoder, device, dtype):
    audio_waveform = audio_processor(audio_waveform, return_tensors="pt", sampling_rate=16000)['input_values']
    audio_waveform = audio_waveform.to(device, dtype)
    audio_embeddings = audio_encoder(audio_waveform).last_hidden_state  # [1, num_embeds, d]

    audio_embeddings = audio_embeddings.permute(1, 0, 2)  # [num_embeds, 1, d]

    return audio_embeddings


# 输入和输出路径
input_dir = '/wangbenyou/huanghj/workspace/research/V-Express/HDTF/short_clip'
output_dir = '/wangbenyou/huanghj/workspace/research/V-Express/HDTF/short_clip_aud_embeds'

# 定义起始文件
start_file = 'WDA_JackReed1_000_clip_0.mp4'
start_processing = False

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 设置设备和数据类型
device = 'cuda:1'
dtype = torch.float32

# 加载Wav2Vec2模型和处理器
audio_encoder_path = './model_ckpts/wav2vec2-base-960h/'
STAN_AUD_FPS = 16000
audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_path).to(dtype=dtype, device=device)
audio_processor = Wav2Vec2Processor.from_pretrained(audio_encoder_path)

# 获取按字典序排序的视频文件
video_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mp4')])

# 遍历视频文件，提取音频并保存嵌入
for video_path in video_files:
    # 检查是否达到了起始文件
    if video_path == start_file:
        start_processing = True

    # 如果还没到起始文件，继续下一个视频
    if not start_processing:
        continue
    
    video_full_path = os.path.join(input_dir, video_path)
    aud_embeds_path = os.path.join(output_dir, video_path.replace('.mp4', '.pt'))

    # 读取视频的音频部分
    _, audio_waveform, meta_info = torchvision.io.read_video(video_full_path, pts_unit='sec')
    audio_sampling_rate = meta_info['audio_fps']
    print(f'Processing {video_path}: Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}.')
    
    # 如果音频采样率不为标准采样率，进行重采样
    if audio_sampling_rate != STAN_AUD_FPS:
        audio_waveform = torchaudio.functional.resample(
            audio_waveform,
            orig_freq=audio_sampling_rate,
            new_freq=STAN_AUD_FPS,
        )
    
    # 取音频的单通道（去掉多通道）
    audio_waveform = audio_waveform.mean(dim=0)

    # 计算音频嵌入
    with torch.no_grad():
        audio_embedding = prepare_audio_embeddings(audio_waveform, audio_processor, audio_encoder, device, dtype)

    # 保存音频嵌入
    torch.save({'global_embeds': audio_embedding}, aud_embeds_path)

    print(f'Saved audio embedding to {aud_embeds_path}')
