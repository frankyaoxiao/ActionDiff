import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import cv2
import imageio
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor

from datasets import ThumosDataset
from torchvision.io.video import read_video
import torch.nn.functional as F

from tqdm import tqdm


def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()

    filter = DeepFloydDataFiltering(verbose=False, device=device)
    return model, filter


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def preprocess(frames):
    frames = frames / 255.0
    frames = 2 * frames - 1
    frames = torch.einsum("t h w c -> t c h w", frames)

    T, C, H, W = frames.shape

    # if H % 64 != 0 or W % 64 != 0:
    #     height, width = map(lambda x: x - x % 64, (H, W))
    #     frames = F.interpolate(frames, size=(height, width), mode='bilinear', align_corners=True)
    #     print(
    #         f"WARNING: Your image is of size {H}x{W} which is not divisible by 64. We are resizing to {height}x{width}!"
    #     )
    frames = F.interpolate(frames, size=(576, 1024), mode='bilinear', align_corners=True)

    assert C == 3, f"expected 3 channels, got {C}"

    # change type to float32
    return frames.float()


def get_svd_latents(model, frames, layer_res, t, resize, context=None):
    t, c, h, w = frames.shape
    return torch.randn(t, 768)


def extract(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    num_frames: Optional[int] = 5,  # 21 for SV3D
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = './results',
    exp_name: str = "generate_vd_feats",
    elevations_deg: Optional[float | List[float]] = 10.0,  # For SV3D
    azimuths_deg: Optional[List[float]] = None,  # For SV3D
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
    dataset: str = "thumos14",
    split: str = "validation",
    layer_res: str = "16x16",
    timestep: int = 0,
    resize: int = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    build_dataset = None
    if dataset == 'thumos14':
        build_dataset = ThumosDataset
        dataset_kwargs = {"split": split}
    else:
        raise ValueError(f"Dataset {dataset} does not exist.")

    dataset = build_dataset(**dataset_kwargs)
    exp_dir = os.path.join(output_folder, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    del model.first_stage_model
    torch.manual_seed(seed)

    with torch.no_grad():
        with torch.autocast(device):
            for idx in tqdm(range(0, len(dataset))):
            # for idx in tqdm(range(0, 2)):
                video_name = dataset.get_meta(idx)['name']
                video_path = dataset.get_meta(idx)['path']
                video, _, _ = read_video(video_path)
                # video = video.to('cuda') # T x H x W x C
                video_feats = []
                feature_stride = 4
                pad_size = num_frames // 2
                offset_frames = 8

                video = F.pad(video, (0, 0, 0, 0, 0, 0, pad_size, pad_size), mode='constant', value=0)

                for frame_idx in tqdm(range(offset_frames + pad_size, len(video) - pad_size, feature_stride)):
                # for frame_idx in tqdm(range(offset_frames + pad_size, offset_frames + pad_size + 2*feature_stride + 1, feature_stride)):
                    frames = video[frame_idx - pad_size : frame_idx + pad_size + 1].to('cuda')
                    # Step 3: Apply inference preprocessing transforms
                    frames = preprocess(frames)

                    value_dict = {
                        "cond_frames_without_noise": frames,
                        "motion_bucket_id": motion_bucket_id,
                        "fps_id": fps_id,
                        "cond_aug": cond_aug,
                        "cond_frames": frames + cond_aug * torch.randn_like(frames)
                    }

                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        [1, num_frames],
                        T=num_frames,
                        device=device,
                    )

                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=[
                            "cond_frames",
                            "cond_frames_without_noise",
                        ],
                    )

                    T, _, H, W = frames.shape
                    F_ = 8
                    C = 4
                    shape = (T, C, H // F_, W // F_)

                    randn = torch.randn(shape, device=device)

                    additional_model_inputs = {}
                    additional_model_inputs["image_only_indicator"] = torch.zeros(
                        2, num_frames
                    ).to(device)
                    additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

                    def denoiser(input, sigma, c,
                                 return_latents=False):
                        return model.denoiser(
                            model.model, input, sigma, c,
                            return_latents=return_latents,
                            **additional_model_inputs
                        )

                    samples_z, latents = model.sampler(denoiser, randn, cond=c, uc=uc,
                                                       return_latents=True, latents_step=23)

                    frame_feats = [latent[num_frames + num_frames//2].mean(dim=(1,2)) for latent in latents]
                    frame_feats = torch.cat(frame_feats)

                    # Step 4: Use the model and print the predicted category
                    # frame_feats = get_svd_latents(model, frames, layer_res, t, resize, context=None).squeeze(0)
                    video_feats.append(frame_feats.detach().cpu().numpy())
                video_feats = np.array(video_feats)
                with open(os.path.join(exp_dir, f"{video_name}.npy"), 'wb') as f:
                    np.save(f, video_feats)

if __name__ == "__main__":
    Fire(extract)
