import torch
import argparse
import os
import time

import accelerate
import cv2
import numpy as np
import torchaudio.functional
import torchaudio
from PIL import Image
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.import_utils import is_xformers_available
from insightface.app import FaceAnalysis
from omegaconf import OmegaConf
from transformers import Wav2Vec2Model, Wav2Vec2Processor, Wav2Vec2FeatureExtractor

from modules import UNet2DConditionModel, UNet3DConditionModel, VKpsGuider, AudioProjection
from pipelines import VExpressPipeline
from pipelines.utils import save_video
from pipelines.utils import retarget_kps
from pipelines.utils import zero_module
from datasets.utils import draw_kps_image
from pipelines.context import compute_context_indices, compute_num_context

# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--unet_config_path', type=str, default='./model_ckpts/stable-diffusion-v1-5/unet/config.json')
    parser.add_argument('--vae_path', type=str, default='./model_ckpts/sd-vae-ft-mse/')
    parser.add_argument('--audio_encoder_path', type=str, default='./model_ckpts/wav2vec2-base-960h/')
    parser.add_argument('--insightface_model_path', type=str, default='./model_ckpts/insightface_models/')

    parser.add_argument('--denoising_unet_path', type=str, default='./model_ckpts/v-express/denoising_unet.bin')
    parser.add_argument('--reference_net_path', type=str, default='./model_ckpts/v-express/reference_net.bin')
    parser.add_argument('--v_kps_guider_path', type=str, default='./model_ckpts/v-express/v_kps_guider.bin')
    parser.add_argument('--audio_projection_path', type=str, default='./model_ckpts/v-express/audio_projection.bin')
    parser.add_argument('--motion_module_path', type=str, default='./model_ckpts/v-express/motion_module.bin')

    parser.add_argument('--retarget_strategy', type=str, default='fix_face',
                        help='{fix_face, no_retarget, offset_retarget, naive_retarget}')

    parser.add_argument('--dtype', type=str, default='fp16')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--do_multi_devices_inference', action='store_true')
    parser.add_argument('--save_gpu_memory', action='store_true')

    parser.add_argument('--num_pad_audio_frames', type=int, default=2)
    parser.add_argument('--standard_audio_sampling_rate', type=int, default=16000)

    parser.add_argument('--reference_image_path', type=str, default='./test_samples/emo/talk_emotion/ref.jpg')
    parser.add_argument('--audio_path', type=str, default='./test_samples/emo/talk_emotion/aud.mp3')
    parser.add_argument('--kps_path', type=str, default='./test_samples/emo/talk_emotion/kps.pth')
    parser.add_argument('--output_path', type=str, default='./output/emo/talk_emotion.mp4')

    parser.add_argument('--test_stage', type=str, default='stage_3')
    parser.add_argument('--audio_embeddings_type', type=str, default='global', help='{global}')

    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--fps', type=float, default=30.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_inference_steps', type=int, default=25)
    parser.add_argument('--guidance_scale', type=float, default=3.5)
    parser.add_argument('--context_frames', type=int, default=24)
    parser.add_argument('--context_overlap', type=int, default=4)
    parser.add_argument('--reference_attention_weight', default=0.95, type=float)
    parser.add_argument('--audio_attention_weight', default=3., type=float)

    args = parser.parse_args()

    return args


def load_reference_net(unet_config_path, reference_net_path, dtype, device):
    reference_net = UNet2DConditionModel.from_config(unet_config_path).to(dtype=dtype, device=device)
    reference_net.load_state_dict(torch.load(reference_net_path, map_location="cpu"), strict=False)
    print(f'Loaded weights of Reference Net from {reference_net_path}.')
    return reference_net


def load_denoising_unet(inf_config_path, unet_config_path, denoising_unet_path, motion_module_path, dtype, device):
    inference_config = OmegaConf.load(inf_config_path)
    denoising_unet = UNet3DConditionModel.from_config_2d(
        unet_config_path,
        unet_additional_kwargs=inference_config.unet_additional_kwargs,
    ).to(dtype=dtype, device=device)
    denoising_unet.load_state_dict(torch.load(denoising_unet_path, map_location="cpu"), strict=False)
    print(f'Loaded weights of Denoising U-Net from {denoising_unet_path}.')

    denoising_unet.load_state_dict(torch.load(motion_module_path, map_location="cpu"), strict=False)
    print(f'Loaded weights of Denoising U-Net Motion Module from {motion_module_path}.')

    return denoising_unet


def load_v_kps_guider(v_kps_guider_path, dtype, device):
    v_kps_guider = VKpsGuider(320, block_out_channels=(16, 32, 96, 256)).to(dtype=dtype, device=device)
    v_kps_guider.load_state_dict(torch.load(v_kps_guider_path, map_location="cpu"))
    print(f'Loaded weights of V-Kps Guider from {v_kps_guider_path}.')
    return v_kps_guider


def load_audio_projection(
        audio_projection_path,
        dtype,
        device,
        inp_dim: int,
        mid_dim: int,
        out_dim: int,
        inp_seq_len: int,
        out_seq_len: int,
):
    audio_projection = AudioProjection(
        dim=mid_dim,
        depth=4,
        dim_head=64,
        heads=12,
        num_queries=out_seq_len,
        embedding_dim=inp_dim,
        output_dim=out_dim,
        ff_mult=4,
        max_seq_len=inp_seq_len,
    ).to(dtype=dtype, device=device)
    audio_projection.load_state_dict(torch.load(audio_projection_path, map_location='cpu'))
    print(f'Loaded weights of Audio Projection from {audio_projection_path}.')
    return audio_projection


def get_scheduler(inference_config_path):
    inference_config = OmegaConf.load(inference_config_path)
    scheduler_kwargs = OmegaConf.to_container(inference_config.noise_scheduler_kwargs)
    scheduler = DDIMScheduler(**scheduler_kwargs)
    return scheduler


def main():
    args = parse_args()
    start_time = time.time()

    if not args.do_multi_devices_inference:
        # TODO
        accelerator = None
        device = torch.device(f'{args.device}:{args.gpu_id}' if args.device == 'cuda' else args.device)
    else:
        accelerator = accelerate.Accelerator()
        device = torch.device(f'cuda:{accelerator.process_index}')
    if args.dtype == 'fp16':
        dtype = torch.float16
    elif args.dtype == 'fp32':
        dtype = torch.float32
    elif args.dtype == 'bf16':
        dtype = torch.bfloat16
    else:
        raise ValueError(f'Do not support {args.dtype}.')

    vae_path = args.vae_path
    audio_encoder_path = args.audio_encoder_path

    vae = AutoencoderKL.from_pretrained(vae_path).to(dtype=dtype, device=device)

    if args.audio_embeddings_type == 'global':
        audio_encoder = Wav2Vec2Model.from_pretrained(audio_encoder_path).to(dtype=dtype, device=device)
        audio_processor = Wav2Vec2Processor.from_pretrained(audio_encoder_path)
    else:
        raise ValueError(f'Do not support {args.audio_embeddings_type}. '
                         f'Now only support "global".')

    unet_config_path = args.unet_config_path
    reference_net_path = args.reference_net_path
    denoising_unet_path = args.denoising_unet_path
    v_kps_guider_path = args.v_kps_guider_path
    audio_projection_path = args.audio_projection_path
    motion_module_path = args.motion_module_path

    inference_config_path = './inference_v2.yaml'
    scheduler = get_scheduler(inference_config_path)
    reference_net = load_reference_net(unet_config_path, reference_net_path, dtype, device)
    denoising_unet = load_denoising_unet(
        inference_config_path, unet_config_path, denoising_unet_path, motion_module_path,
        dtype, device
    )
    v_kps_guider = load_v_kps_guider(v_kps_guider_path, dtype, device)

    if args.audio_embeddings_type == 'global':
        inp_dim = 768
    else:
        raise ValueError(f'Do not support {args.audio_embeddings_type}. '
                         f'Now only support "global".')
    audio_projection = load_audio_projection(
        audio_projection_path,
        dtype,
        device,
        inp_dim=inp_dim,
        mid_dim=denoising_unet.config.cross_attention_dim,
        out_dim=denoising_unet.config.cross_attention_dim,
        inp_seq_len=2 * (2 * args.num_pad_audio_frames + 1),
        out_seq_len=2 * args.num_pad_audio_frames + 1,
    )

    if args.test_stage == 'stage_1':
        for k, v in denoising_unet.named_parameters():
            if 'temporal_transformer.proj_out' in k:
                zero_module(v)
            if 'attn2.to_out' in k:
                zero_module(v)
    elif args.test_stage == 'stage_2':
        pass
    elif args.test_stage == 'stage_3':
        pass
    else:
        raise NotImplementedError(f"{args.test_stage}")

    generator = torch.manual_seed(args.seed)
    pipeline = VExpressPipeline(
        vae=vae,
        reference_net=reference_net,
        denoising_unet=denoising_unet,
        v_kps_guider=v_kps_guider,
        audio_processor=audio_processor,
        audio_encoder=audio_encoder,
        audio_projection=audio_projection,
        scheduler=scheduler,
    ).to(dtype=dtype, device=device)

    app = FaceAnalysis(
        providers=['CUDAExecutionProvider' if args.device == 'cuda' else 'CPUExecutionProvider'],
        provider_options=[{'device_id': args.gpu_id}] if args.device == 'cuda' else [],
        root=args.insightface_model_path,
    )
    app.prepare(ctx_id=0, det_size=(args.image_height, args.image_width))

    reference_image = Image.open(args.reference_image_path).convert('RGB')
    # reference_image = reference_image.resize((args.image_height, args.image_width))

    reference_image_for_kps = cv2.imread(args.reference_image_path)
    reference_image_for_kps = cv2.resize(reference_image_for_kps, (args.image_width, args.image_height))
    reference_kps = app.get(reference_image_for_kps)[0].kps[:3]
    if args.save_gpu_memory:
        del app
    torch.cuda.empty_cache()

    audio_waveform, audio_sampling_rate = torchaudio.load(args.audio_path)
    print(f'Length of audio is {audio_waveform.shape[1]} with the sampling rate of {audio_sampling_rate}.')
    if audio_sampling_rate != args.standard_audio_sampling_rate:
        audio_waveform = torchaudio.functional.resample(
            audio_waveform,
            orig_freq=audio_sampling_rate,
            new_freq=args.standard_audio_sampling_rate,
        )
    audio_waveform = audio_waveform.mean(dim=0)

    duration = audio_waveform.shape[0] / args.standard_audio_sampling_rate
    init_video_length = int(duration * args.fps)

    num_contexts = compute_num_context(init_video_length, args.context_frames, args.context_overlap)
    context_indices = compute_context_indices(
        num_context=num_contexts, context_size=args.context_frames, context_overlap=args.context_overlap
    )
    video_length = context_indices[-1][1] + 1
    fps = video_length / duration
    print(f'The corresponding video length is {video_length}. fps: {fps}')

    kps_sequence = None
    if args.kps_path != "":
        assert os.path.exists(args.kps_path), f'{args.kps_path} does not exist'
        kps_sequence = torch.tensor(torch.load(args.kps_path))  # [len, 3, 2]
        print(f'The original length of kps sequence is {kps_sequence.shape[0]}.')

        if kps_sequence.shape[0] > video_length:
            kps_sequence = kps_sequence[:video_length, :, :]

        kps_sequence = torch.nn.functional.interpolate(kps_sequence.permute(1, 2, 0), size=video_length, mode='linear')
        kps_sequence = kps_sequence.permute(2, 0, 1)
        print(f'The interpolated length of kps sequence is {kps_sequence.shape[0]}.')

    retarget_strategy = args.retarget_strategy
    if retarget_strategy == 'fix_face':
        kps_sequence = torch.tensor([reference_kps] * video_length)
    elif retarget_strategy == 'no_retarget':
        kps_sequence = kps_sequence
    elif retarget_strategy == 'offset_retarget':
        kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=True)
    elif retarget_strategy == 'naive_retarget':
        kps_sequence = retarget_kps(reference_kps, kps_sequence, only_offset=False)
    else:
        raise ValueError(f'The retarget strategy {retarget_strategy} is not supported.')

    kps_images = []
    for i in range(video_length):
        kps_image = draw_kps_image(args.image_height, args.image_width, kps_sequence[i])
        kps_images.append(Image.fromarray(kps_image))

    video_tensor = pipeline(
        reference_image=reference_image,
        kps_images=kps_images,
        audio_waveform=audio_waveform,
        width=args.image_width,
        height=args.image_height,
        video_length=video_length,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        context_frames=args.context_frames,
        context_overlap=args.context_overlap,
        reference_attention_weight=args.reference_attention_weight,
        audio_attention_weight=args.audio_attention_weight,
        num_pad_audio_frames=args.num_pad_audio_frames,
        generator=generator,
        do_multi_devices_inference=args.do_multi_devices_inference,
        save_gpu_memory=args.save_gpu_memory,
    )

    if accelerator is None or accelerator.is_main_process:
        save_video(video_tensor, args.audio_path, args.output_path, device, fps)
        consumed_time = time.time() - start_time
        generation_fps = video_tensor.shape[2] / consumed_time
        print(f'The generated video has been saved at {args.output_path}. '
              f'The generation time is {consumed_time:.1f} seconds. '
              f'The generation FPS is {generation_fps:.2f}.')


if __name__ == '__main__':
    main()
