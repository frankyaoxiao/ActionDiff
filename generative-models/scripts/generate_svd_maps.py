import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import List, Optional
import time

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
import torch, torchvision
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.inference.helpers import embed_watermark
from sgm.util import default, instantiate_from_config
from torchvision.transforms import ToTensor

from datasets import (ThumosDataset, AnimalKingdomActionRecognitionDataset, CharadesDataset, PhavDataset,
                      MiniSportsDataset, UCFCrimeDataset, XDViolenceDataset, UCFDataset, HMDBDataset,
                      EpicKitchensDataset)
from torchvision.io.video import read_video
import torch.nn.functional as F
from sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder2

from tqdm import tqdm

from sgm.modules.encoders.modules import FrozenOpenCLIPEmbedder


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


def extract(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    input_type: str = "video",  # Either "video" or "frames"
    num_frames: Optional[int] = 3,  # 21 for SV3D
    feature_stride: int = 4,
    offset_frames: int = 8,
    num_steps: Optional[int] = None,
    version: str = "svd",
    fps_id: int = 6,
    motion_bucket_id: int = 127,
    cond_aug: float = 0.02,
    seed: int = 23,
    decoding_t: int = 14,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = './results',
    exp_name: str = "generate_svd_feats",
    elevations_deg: Optional[float | List[float]] = 10.0,  # For SV3D
    azimuths_deg: Optional[List[float]] = None,  # For SV3D
    image_frame_ratio: Optional[float] = None,
    verbose: Optional[bool] = False,
    dataset: str = "thumos14",
    split: str = None,
    layer_res: str = "16x16",
    timestep: int = 0,
    resize: int = None,
    reduction: str = None,
    layer_idxs: List[int] = None,
    conditioning: str = None,
    start_idx: int = None,
    end_idx: int = None,
    diffusion_step: int = 23, # between 0 (0%) and 24 (100%)
    add_noise: bool = True,
    shuffle: bool = False,
    show_time = False,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    Args:
        input_type: Either "video" for video files or "frames" for sequences of frames
    """

    if layer_idxs is None:
        layer_idxs = list(range(1, 14))
    elif isinstance(layer_idxs, int):
        layer_idxs = [layer_idxs]

    if version == "svd":
        num_frames = default(num_frames, 14)
        num_steps = default(num_steps, 25)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd/")
        model_config = "scripts/sampling/configs/svd.yaml"
    elif version == "svd_xt":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt.yaml"
    elif version == "sd_v2":
        num_frames = default(num_frames, 25)
        num_steps = default(num_steps, 30)
        output_folder = default(output_folder, "outputs/simple_video_sample/svd_xt/")
        model_config = "scripts/sampling/configs/svd_xt_sdv2.yaml"
    else:
        raise ValueError(f"Version {version} does not exist.")

    print(f"Conditioning: {conditioning}")
    print(f"Diffusion step: {diffusion_step}/{num_steps}")
    print(f"Add noise: {add_noise}")
    print(f"Layer indices: {layer_idxs}")
    print(f"Version: {version}")
    print(f"Dataset: {dataset}")

    build_dataset = None
    if dataset == 'thumos14':
        build_dataset = ThumosDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'animal_kingdom':
        build_dataset = AnimalKingdomActionRecognitionDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'charades':
        build_dataset = CharadesDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'phav':
        build_dataset = PhavDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'minisports':
        build_dataset = MiniSportsDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'ucfcrime':
        build_dataset = UCFCrimeDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'xdviolence':
        build_dataset = XDViolenceDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'ucf':
        build_dataset = UCFDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'hmdb':
        build_dataset = HMDBDataset
        dataset_kwargs = {"split": split}
    elif dataset == 'epic_kitchens':
        build_dataset = EpicKitchensDataset
        dataset_kwargs = {"split": split}
    else:
        raise ValueError(f"Dataset {dataset} does not exist.")

    dataset = build_dataset(**dataset_kwargs)
    exp_dir = os.path.join(output_folder, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    feats_dir = (os.path.join(exp_dir, f"feats_{reduction}") if reduction is not None
                 else os.path.join(exp_dir, "feats"))
    os.makedirs(os.path.join(feats_dir), exist_ok=True)
    if reduction is None:
        for layer_idx in layer_idxs:
            os.makedirs(os.path.join(feats_dir, f"layer_{layer_idx}"), exist_ok=True)
    # os.makedirs(os.path.join(exp_dir, 'feats_max'), exist_ok=True)

    model, filter = load_model(
        model_config,
        device,
        num_frames,
        num_steps,
        verbose,
    )
    del model.first_stage_model.decoder
    torch.cuda.empty_cache()
    # del model.first_stage_model
    torch.manual_seed(seed)

    if conditioning == "actions":
        cond_embedding = []
        cond_model = FrozenOpenCLIPEmbedder2(always_return_pooled=True, legacy=False).to('cuda')
        for action in dataset.actions:
            _, action_feats = cond_model(action)
            cond_embedding.append(action_feats)
        cond_embedding = torch.cat(cond_embedding)
        cond_embedding = cond_embedding.expand(num_frames, -1, -1)

        _, uncond_embedding = cond_model("")
        uncond_embedding = uncond_embedding.expand(num_frames, len(dataset.actions), -1)

        del cond_model

    elif conditioning == "empty_str":
        cond_model = FrozenOpenCLIPEmbedder2(always_return_pooled=True, legacy=False).to('cuda')
        _, uncond_embedding = cond_model("")
        uncond_embedding = uncond_embedding.expand(num_frames, len(dataset.actions), -1)
        cond_embedding = uncond_embedding

        del cond_model

    print(f"Dataset length: {len(dataset)} videos")

    if start_idx is None:
        start_idx = 0

    if end_idx is None:
        end_idx = len(dataset)

    dataset_start_time = time.time()
    dataset_n_frmaes = 0

    with (torch.no_grad()):
        with torch.autocast(device):
            # for idx in tqdm(range(7 * (end_idx // 8), end_idx)):
            idx_range = list(range(start_idx, end_idx))
            if shuffle:
                np.random.shuffle(idx_range)
            for idx in tqdm(idx_range):

                video_start_time = time.time()

                # Recovery
                if idx < 0:
                    continue
                
                video_name = dataset.get_meta(idx)['name']
                video_path = dataset.get_meta(idx)['path']
                video_kwargs = None
                if "kwargs" in dataset.get_meta(idx):
                    video_kwargs = dataset.get_meta(idx)["kwargs"]
                
                # Check if features already exist for all layers for recovery purposes
                all_features_exist = True
                for layer_idx in layer_idxs:
                    feature_path = os.path.join(feats_dir, f"layer_{layer_idx}", f"{video_name}_mean.npy")
                    if not os.path.exists(feature_path):
                        all_features_exist = False
                        break
                
                if all_features_exist:
                    continue
                
                if input_type == "video":
                    # print(video_path)
                    if video_kwargs is not None:
                        video, _, _ = read_video(video_path, **video_kwargs)
                    else:
                        video, _, _ = read_video(video_path, start_pts=0, end_pts=180, pts_unit='sec')
                    # print(video.shape)
                elif input_type == "frames":
                    frame_files = sorted(glob(os.path.join(video_path, "*.jpg")))  # Adjust extension if needed
                    frames = []
                    for frame_file in frame_files:
                        frame = torchvision.io.read_image(frame_file)  # C x H x W
                        frame = frame.permute(1, 2, 0)  # H x W x C
                        frames.append(frame)
                    video = torch.stack(frames, dim=0)  # T x H x W x C
                else:
                    raise ValueError(f"Invalid input_type: {input_type}. Must be either 'video' or 'frames'")

                video_feats = {layer_idx: [] for layer_idx in layer_idxs}
                video_feats_mean = {layer_idx: [] for layer_idx in layer_idxs}

                T, H, W, C = video.shape

                video = video[offset_frames:T:feature_stride]

                pad_size = (num_frames - (len(video) % num_frames)) % num_frames

                video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, pad_size), mode='constant', value=0)

                for frame_idx in range(0, len(video), num_frames):
                # for frame_idx in tqdm(range(0, len(video), num_frames)):
                # for frame_idx in tqdm(range(offset_frames + pad_size, offset_frames + pad_size + 2*feature_stride + 1, feature_stride)):
                    frames = video[frame_idx:frame_idx+num_frames].to('cuda')
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

                    if conditioning == 'no_cond':
                        c = uc
                    elif conditioning in ['actions', 'empty_str']:
                        c['crossattn'] = cond_embedding
                        uc['crossattn'] = uncond_embedding
                    elif conditioning is not None:
                        raise ValueError(f"Conditioning {conditioning} does not exist.")

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

                    # samples_z, latents = model.sampler(denoiser, randn, cond=c, uc=uc,
                    #                                    return_latents=True, latents_step=23)

                    latents = get_svd_latents(frames=frames,
                                              model=model,
                                              denoiser=denoiser,
                                              cond=c,
                                              uc=uc,
                                              num_steps=num_steps,
                                              diff_step=diffusion_step,
                                              add_noise=add_noise)

                    if pad_size > 0 and (frame_idx + num_frames > len(video) - pad_size):
                        latents = [latent[:-pad_size] for latent in latents]

                    # if reduction == "mean":
                    #     frame_feats = [latent[num_frames:].mean(dim=(-1,-2)) for latent in latents]
                    #
                    #     # Step 4: Use the model and print the predicted category
                    #     # frame_feats = get_svd_latents(model, frames, layer_res, t, resize, context=None).squeeze(0)
                    #     # video_feats.append(frame_feats_mean.detach().cpu().numpy())
                    #
                    # elif reduction == "max":
                    #     frame_feats = [latent[num_frames:].max(dim=-1)[0].max(dim=-1)[0] for latent in latents]
                    #
                    # elif reduction is None:
                    if reduction is None:
                        for layer_idx in layer_idxs:
                            frame_feats = latents[layer_idx - 1][num_frames:]
                            frame_feats_mean = torch.mean(frame_feats, dim=(-1,-2))
                            # frame_feats = F.interpolate(frame_feats, size=(3, 5), mode='bilinear', align_corners=True)
                            # frame_feats = torch.permute(frame_feats, (0, 2, 3, 1))

                            # # Step 4: Use the model and print the predicted category
                            # # frame_feats = get_svd_latents(model, frames, layer_res, t, resize, context=None).squeeze(0)

                            # video_feats[layer_idx].append(frame_feats.detach().cpu().numpy())
                            video_feats_mean[layer_idx].append(frame_feats_mean.detach().cpu().numpy())

                for layer_idx in layer_idxs:
                    # video_feats[layer_idx] = np.concatenate(video_feats[layer_idx], axis=0)
                    video_feats_mean[layer_idx] = np.concatenate(video_feats_mean[layer_idx], axis=0)

                    # with open(os.path.join(feats_dir, f"layer_{layer_idx}", f"{video_name}.npy"), 'wb') as f:
                    #     np.save(f, video_feats)

                    with open(os.path.join(feats_dir, f"layer_{layer_idx}", f"{video_name}_mean.npy"), 'wb') as f:
                        np.save(f, video_feats_mean[layer_idx])

                video_end_time = time.time()
                video_n_frames = len(video)
                print(f"Video frames/sec = {video_n_frames / (video_end_time - video_start_time):.2f}")

                dataset_n_frmaes += video_n_frames
                print(f"Dataset frames/sec = {dataset_n_frmaes / (video_end_time - dataset_start_time):.2f}")


def to_str(layer_idxs):
    return '_'.join(map(str, layer_idxs))


def get_svd_latents(frames, model, denoiser, cond, uc, num_steps, diff_step, add_noise):

    z = model.encode_first_stage(frames)

    _, s_in, sigmas, num_sigmas, cond, uc = prepare_extraction_loop(
        model.sampler, z, cond, uc, num_steps
    )

    # schedule = model.sampler.get_sigma_gen(num_sigmas)
    #
    # t = schedule[diff_step]

    t = diff_step

    gamma = (
        min(model.sampler.s_churn / (num_sigmas - 1), 2 ** 0.5 - 1)
        if model.sampler.s_tmin <= sigmas[t] <= model.sampler.s_tmax
        else 0.0
    )

    if not add_noise:
        gamma = 0.0
        sigmas[t] = 0.0

    _, latents = model.sampler.sampler_step(
        s_in * sigmas[t],
        None, # s_in * sigmas[t + 1],
        denoiser,
        z,
        cond,
        uc,
        gamma,
        return_latents=True,
    )

    return latents


def prepare_extraction_loop(self, x, cond, uc=None, num_steps=None):
    sigmas = self.discretization(
        self.num_steps if num_steps is None else num_steps, device=self.device
    )
    uc = default(uc, cond)

    # x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
    num_sigmas = len(sigmas)

    s_in = x.new_ones([x.shape[0]])

    return x, s_in, sigmas, num_sigmas, cond, uc


if __name__ == "__main__":
    Fire(extract)
