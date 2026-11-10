#!/usr/bin/env python3
"""Checkpoint -> distributed sampling -> NPZ -> ADM FID."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from assets import prepare_metric_assets, prepare_vae, prepare_stats, sha256

ROOT = Path(__file__).resolve().parent


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['prepare', 'run', 'fid'])
    p.add_argument('--cache-dir', default=str(ROOT / 'assets'))
    p.add_argument('--offline', action='store_true')
    p.add_argument('--ref-batch', help='Override reference NPZ (must match dataset/resolution)')
    p.add_argument('--inception', help='Override ADM classify_image_graph_def.pb')
    p.add_argument('--resolution', type=int, choices=[256, 512], default=None)
    p.add_argument('--vae', choices=['mse', 'ema'], default='mse')
    p.add_argument('--latent-stats', help='Training latents_scale/latents_bias .pt file')
    p.add_argument('--ckpt', help='SIT checkpoint containing ema state dict and optional args')
    p.add_argument('--model', choices=['SiT-S/2', 'SiT-B/2', 'SiT-L/2', 'SiT-XL/2'])
    p.add_argument('--qk-norm', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--path-type', choices=['linear', 'cosine'], default=None)
    p.add_argument('--gpus', default='0', help='CUDA_VISIBLE_DEVICES, e.g. 0,1')
    p.add_argument('--batch-size', type=int, default=32, help='Samples per GPU')
    p.add_argument('--fid-python', default=sys.executable)
    p.add_argument('--fid-batch-size', type=int, default=64)
    p.add_argument('--num-samples', type=int, default=50000)
    p.add_argument('--num-steps', type=int, default=250)
    p.add_argument('--mode', choices=['sde', 'ode'], default='sde')
    p.add_argument('--cfg-scale', type=float, default=1.0)
    p.add_argument('--guidance-low', type=float, default=0.0)
    p.add_argument('--guidance-high', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--inference-dtype', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    p.add_argument('--tf32', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--fused-attn', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--heun', action=argparse.BooleanOptionalAction, default=False)
    p.add_argument('--output-dir', default='fid-output')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--samples', help='Existing NPZ for the fid command')
    p.add_argument('--dist-timeout-minutes', type=int, default=180)
    return p


def resolve_checkpoint(args):
    import torch
    args.ckpt = str(Path(args.ckpt).expanduser().resolve())
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    if 'ema' not in ckpt:
        raise ValueError('Expected a SIT training checkpoint with an ema state dict')
    meta = ckpt.get('args')
    def saved(name, default):
        return meta.get(name, default) if isinstance(meta, dict) else getattr(meta, name, default)
    if args.model is None:
        args.model = saved('model', '').replace('-EncoderKV', '')
        if args.model not in ['SiT-S/2', 'SiT-B/2', 'SiT-L/2', 'SiT-XL/2']:
            raise ValueError('Cannot infer model; supply --model SiT-XL/2 (or the correct architecture)')
    for name, default in [('resolution', 256), ('qk_norm', False), ('path_type', 'linear')]:
        if getattr(args, name) is None:
            setattr(args, name, saved(name, default))
    if args.resolution not in (256, 512) or saved('num_classes', 1000) != 1000:
        raise ValueError('This bundle supports ImageNet 1000 classes at 256/512 only')
    if args.path_type not in ('linear', 'cosine'):
        raise ValueError('Unsupported checkpoint path_type')


def launch(command, log, env):
    print('Launching: ' + ' '.join(map(str, command)), flush=True)
    with log.open('a') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                stream.write(line)
                stream.flush()
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, command)


def sampling_command(args, stats, vae, output):
    nproc = len(args.gpus.split(','))
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f'--nproc_per_node={nproc}', str(ROOT / 'generate.py')]
    values = {'ckpt': args.ckpt, 'model': args.model, 'resolution': args.resolution,
              'latent-stats': stats, 'vae-dir': vae, 'output-folder': output / 'samples',
              'vae': args.vae, 'per-proc-batch-size': args.batch_size,
              'num-fid-samples': args.num_samples, 'num-steps': args.num_steps,
              'mode': args.mode, 'cfg-scale': args.cfg_scale,
              'guidance-low': args.guidance_low, 'guidance-high': args.guidance_high,
              'global-seed': args.seed, 'inference-dtype': args.inference_dtype,
              'path-type': args.path_type, 'dist-timeout-minutes': args.dist_timeout_minutes}
    for name, value in values.items():
        command += ['--' + name, str(value)]
    for name in ['qk_norm', 'tf32', 'fused_attn', 'heun']:
        command.append('--' + ('' if getattr(args, name) else 'no-') + name.replace('_', '-'))
    if args.resume:
        command.append('--resume-samples')
    return command


def main():
    p = parser()
    args = p.parse_args()
    if min(args.batch_size, args.fid_batch_size, args.num_steps, args.dist_timeout_minutes) < 1 or args.num_samples < 2:
        p.error('batch sizes, num-steps, timeout must be positive; num-samples >= 2')
    if not 0 <= args.guidance_low <= args.guidance_high <= 1 or args.cfg_scale < 1:
        p.error('Require 0 <= guidance-low <= guidance-high <= 1 and cfg-scale >= 1')
    if not args.gpus or any(not g.strip().isdigit() for g in args.gpus.split(',')):
        p.error('--gpus must be comma-separated integer GPU IDs')
    if len(set(args.gpus.split(','))) != len(args.gpus.split(',')):
        p.error('Duplicate GPU IDs')
    if args.command == 'run':
        if not args.ckpt:
            p.error('run requires --ckpt')
        # Fail before a long sampling run if the selected FID environment is incomplete.
        subprocess.run([args.fid_python, '-c',
                        'import tensorflow.compat.v1, scipy, numpy, requests, tqdm'], check=True)
        resolve_checkpoint(args)
    else:
        args.resolution = args.resolution or 256
    cache = Path(args.cache_dir).expanduser().resolve()
    stats = prepare_stats(args) if args.command != 'fid' else None
    if args.command == 'fid':
        if not args.samples:
            p.error('fid requires --samples')
        args.samples = str(Path(args.samples).expanduser().resolve())
        from fid import inspect_samples
        inspect_samples(args.samples, args.resolution)
    ref, inception = prepare_metric_assets(args)
    if args.command == 'prepare':
        vae = prepare_vae(args)
        print(f'Assets ready: {cache}\nVAE: {vae}\nLatent stats: {stats}')
        return
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.offline:
        env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    if args.command == 'run':
        vae = prepare_vae(args)
        command = sampling_command(args, stats, vae, output)
        contract = {'sampling_command': [x for x in command if x != '--resume-samples'],
                    'gpus': args.gpus, 'checkpoint_sha256': sha256(args.ckpt),
                    'latent_stats_sha256': sha256(stats),
                    'vae_sha256': ({vae.name: sha256(vae)} if vae.is_file() else
                                   {str(f.relative_to(vae)): sha256(f) for f in sorted(vae.rglob('*')) if f.is_file()}),
                    'code_sha256': {f: sha256(ROOT / f) for f in ['generate.py', 'samplers.py', 'models/sit.py', 'models/autoencoder.py']}}
        manifest = output / 'sampling.json'
        if manifest.exists():
            if not args.resume:
                p.error('Output already used; choose another --output-dir or pass --resume')
            if json.loads(manifest.read_text()) != contract:
                p.error('Resume settings/weights/code changed; use a new output directory')
        else:
            if (output / 'samples').exists() or (output / 'samples.npz').exists():
                p.error('Samples exist without sampling.json; use a new output directory')
            manifest.write_text(json.dumps(contract, indent=2) + '\n')
        launch(command, output / 'sampling.log', env)
        args.samples = str(output / 'samples.npz')
    fid_command = [args.fid_python, str(ROOT / 'fid.py'), '--samples', args.samples,
                   '--ref-batch', str(ref), '--inception', str(inception),
                   '--resolution', str(args.resolution), '--batch-size', str(args.fid_batch_size),
                   '--output', str(output / 'fid.json')]
    if args.command == 'run':
        fid_command += ['--expected-count', str(args.num_samples)]
    cpu_marker = ROOT / '.fid-cpu'
    if cpu_marker.is_file() and cpu_marker.read_text().strip() == '1':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    launch(fid_command, output / 'fid.log', env)


if __name__ == '__main__':
    main()
