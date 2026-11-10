# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples images from a trained SiT-DINOKV model using the original SiT architecture.
This script loads DINOKV checkpoints but uses sit.py for inference.
The DINO-KV specific weights are ignored via strict=False loading.
"""
import torch
import torch.distributed as dist
from models.sit import SiT_models
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
import datetime
from diffusers.models import AutoencoderKL
from models.autoencoder import VAE_F8D4
from samplers import euler_sampler, euler_maruyama_sampler


def valid_sample(path, resolution):
    """A killed writer may leave a truncated PNG; those must be regenerated."""
    try:
        with Image.open(path) as img:
            if img.size != (resolution, resolution) or img.mode != "RGB":
                return False
            img.load()
        return True
    except (OSError, ValueError):
        return False


def advance_sampler_rng(latents, mode, num_steps):
    """Replay the random draws in samplers.py without model inference.

    SDE draws float64 noise once per non-final step; ODE draws none.
    The caller has already drawn the initial latents and class labels.
    Keep this in sync with euler_maruyama_sampler when changing that sampler.
    """
    if mode == "sde":
        noise_template = latents.to(torch.float64)
        for _ in range(num_steps - 1):
            torch.randn_like(noise_template)


def create_npz_from_sample_folder(sample_dir, num=50_000):
    # Stream PNG bytes into an uncompressed NPZ; no 10-20 GB stacking allocation.
    import zipfile
    target = str(sample_dir) + ".npz"
    temporary = target + ".partial"
    with Image.open(f"{sample_dir}/000000.png") as first:
        resolution = first.width
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
        with archive.open("arr_0.npy", "w", force_zip64=True) as output:
            np.lib.format.write_array_header_2_0(output, {
                "descr": "|u1", "fortran_order": False,
                "shape": (num, resolution, resolution, 3),
            })
            for i in tqdm(range(num), desc="Packing NPZ"):
                path = f"{sample_dir}/{i:06d}.png"
                if not valid_sample(path, resolution):
                    raise ValueError(f"Invalid PNG: {path}")
                with Image.open(path) as image:
                    output.write(np.asarray(image, dtype=np.uint8).tobytes())
    os.replace(temporary, target)
    return target


def load_vae(device, args):
    stats = torch.load(args.latent_stats, map_location="cpu", weights_only=False)
    scale = stats["latents_scale"].to(device).view(1, -1, 1, 1)
    bias = stats["latents_bias"].to(device).view(1, -1, 1, 1)
    if scale.numel() not in (1, 4) or bias.numel() not in (1, 4):
        raise ValueError("Latent stats must be scalar or have four channels")
    if not torch.isfinite(scale).all() or not torch.isfinite(bias).all() or (scale == 0).any():
        raise ValueError("Invalid latent stats")
    if os.path.isfile(args.vae_dir):
        vae = VAE_F8D4().to(device).eval()
        vae.load_state_dict(torch.load(args.vae_dir, map_location="cpu", weights_only=False))
    else:
        vae = AutoencoderKL.from_pretrained(args.vae_dir, local_files_only=True).to(device).eval()
    vae.requires_grad_(False)
    return vae, scale, bias


def main(args):
    """Run sampling."""
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling requires GPU"
    torch.set_grad_enabled(False)

    # Setup DDP
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=args.dist_timeout_minutes))
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{device_id}")
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device_id)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.inference_dtype == "fp32":
        inference_dtype = torch.float32
    elif args.inference_dtype == "bf16":
        inference_dtype = torch.bfloat16
    elif args.inference_dtype == "fp16":
        inference_dtype = torch.float16
    else:
        raise ValueError(f"Unknown inference dtype: {args.inference_dtype}")

    # Load checkpoint and extract saved args for auto-config
    ckpt_path = args.ckpt
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', None)

    # Load model (using original SiT)
    block_kwargs = {"fused_attn": args.fused_attn, "qk_norm": args.qk_norm}
    latent_size = args.resolution // 8

    # Note: encoder_depth and z_dims are not needed at inference time (eval_mode=True)
    # because REPA projections are skipped in eval mode
    model = SiT_models[args.model](
        input_size=latent_size,
        in_channels=4,
        num_classes=args.num_classes,
        use_cfg=True,
        eval_mode=True,
        **block_kwargs,
    ).to(device=device, dtype=inference_dtype)

    # Load DINOKV checkpoint with strict=False (ignores DINO-KV specific weights)
    state_dict = ckpt['ema']

    # Filter out DINO-KV specific keys
    filtered_state_dict = {}
    ignored_keys = []
    for k, v in state_dict.items():
        if (
            'kv_proj' in k or 'residual_proj' in k or 'hidden_proj' in k
            or '.attn.g_k.' in k or '.attn.g_v.' in k
        ):  # Training-only scaffold projections
            ignored_keys.append(k)
        else:
            filtered_state_dict[k] = v

    if rank == 0 and ignored_keys:
        print(f"Ignored {len(ignored_keys)} DINO-KV specific keys")

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    if rank == 0:
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    required_missing = [k for k in missing if not k.startswith("projectors.")]
    if required_missing:
        raise ValueError(f"Checkpoint missing inference weights: {required_missing}")
    model.eval()
    if rank == 0:
        print(f"Inference dtype: {args.inference_dtype}")

    # Load VAE
    vae, latents_scale, latents_bias = load_vae(device, args)

    assert args.cfg_scale >= 1.0, "cfg_scale should be >= 1.0"

    # Create output folder
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    cfg_intv = "" if args.guidance_low == 0. and args.guidance_high == 1. else f"_{args.guidance_low}_{args.guidance_high}"
    dtype_suffix = "" if args.inference_dtype == "fp32" else f"-{args.inference_dtype}"
    hparams = f"vae{args.vae}-cfg{args.cfg_scale}{cfg_intv}-seed{args.global_seed}-mode{args.mode}{dtype_suffix}-steps{args.num_steps}_{ckpt_string_name}"
    exp_name = os.path.basename(os.path.normpath(args.ckpt.rsplit("checkpoints")[0]))
    sample_folder_dir = args.output_folder
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # Sampling loop
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total images to sample: {total_samples}")
        print(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total_generated = 0
    skipped = 0
    written = 0
    if args.resume_samples:
        print(f"[rank {rank}] Resuming PNGs: keep checkpoint, GPU/rank order, world size, "
              "batch size, seed and all sampling settings identical to the original run.", flush=True)

    for _ in pbar:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device, dtype=inference_dtype)
        y = torch.randint(0, args.num_classes, (n,), device=device)
        paths = [os.path.join(sample_folder_dir,
                 f"{(total_generated + i) * dist.get_world_size() + rank:06d}.png")
                 for i in range(n)]
        existing = [args.resume_samples and valid_sample(path, args.resolution)
                    for path in paths]
        if all(existing):
            # Merely continuing here would reuse early RNG states for later images.
            advance_sampler_rng(z, args.mode, args.num_steps)
            skipped += n
            total_generated += n
            continue

        sampling_kwargs = dict(
            model=model,
            latents=z,
            y=y,
            num_steps=args.num_steps,
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=args.path_type,
        )

        with torch.no_grad():
            if args.mode == "sde":
                samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
            elif args.mode == "ode":
                samples = euler_sampler(**sampling_kwargs).to(torch.float32)
            else:
                raise NotImplementedError()

            samples = vae.decode(samples / latents_scale + latents_bias).sample
            samples = (samples + 1) / 2.
            samples = torch.clamp(255. * samples, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            for i, sample in enumerate(samples):
                if existing[i]:
                    skipped += 1
                    continue
                # Rename only after a complete PNG has been written.
                temp_path = paths[i] + ".tmp"
                Image.fromarray(sample).save(temp_path, format="PNG")
                os.replace(temp_path, paths[i])
                written += 1

        total_generated += n

    print(f"[rank {rank}] Sampling complete: kept={skipped}, written={written}; "
          "waiting for other ranks.", flush=True)
    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-stats", required=True)
    parser.add_argument("--vae-dir", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference-dtype", type=str, default="fp32", choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--resume-samples", action="store_true",
                        help="Keep valid existing PNGs; requires the original rank layout and sampling settings")
    parser.add_argument("--dist-timeout-minutes", type=int, default=180,
                        help="Collective timeout, including waiting for slower sampling ranks")

    # Model (using original SiT names)
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=256)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)

    # Sampling
    parser.add_argument("--per-proc-batch-size", type=int, default=256)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--mode", type=str, default="sde")
    parser.add_argument("--vae", type=str, choices=["mse", "ema"], default="mse")
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--path-type", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--guidance-low", type=float, default=0.)
    parser.add_argument("--guidance-high", type=float, default=1.)

    args = parser.parse_args()
    main(args)
