"""Asset downloads do not require torch or TensorFlow."""
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

BASE = 'https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/'
INCEPTION = BASE + 'classify_image_graph_def.pb'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, target, offline=False, expected_sha256=None):
    target = Path(target)
    receipt = target.with_name(target.name + '.download.json')
    if target.is_file() and receipt.is_file():
        info = json.loads(receipt.read_text())
        if (info['url'] == url and target.stat().st_size == info['size']
                and sha256(target) == info['sha256']
                and (expected_sha256 is None or info['sha256'] == expected_sha256)):
            print(f'Using verified cache: {target}', flush=True)
            return target
        raise ValueError(f'Corrupt cache: {target}; move/delete this file and its .download.json, then retry')
    if offline:
        raise FileNotFoundError(f'Missing verified offline asset: {target}; run prepare online first')
    if target.exists():
        raise ValueError(f'Unverified existing asset: {target}; use --ref-batch/--inception for manually downloaded files')
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + '.partial')
    for attempt in range(3):
        try:
            print(f'Downloading {url} -> {target}', flush=True)
            with urllib.request.urlopen(url, timeout=60) as response, part.open('wb') as f:
                expected = int(response.headers.get('Content-Length', 0))
                count = 0
                last = time.monotonic()
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    count += len(chunk)
                    if time.monotonic() - last > 5:
                        print(f'  {count / 2**20:.0f} / {expected / 2**20:.0f} MiB', flush=True)
                        last = time.monotonic()
            if count == 0 or (expected and count != expected):
                raise IOError(f'Incomplete download: {count}/{expected} bytes')
            info = {'url': url, 'size': count, 'sha256': sha256(part)}
            if expected_sha256 and info['sha256'] != expected_sha256:
                raise ValueError(f'Official SHA256 mismatch: {target}')
            os.replace(part, target)
            receipt.write_text(json.dumps(info, indent=2) + '\n')
            return target
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def prepare_metric_assets(args):
    cache = Path(args.cache_dir).expanduser().resolve()
    name = 'VIRTUAL_imagenet256_labeled.npz' if args.resolution == 256 else 'VIRTUAL_imagenet512.npz'
    ref = Path(args.ref_batch).expanduser().resolve() if args.ref_batch else download(
        BASE + f'imagenet/{args.resolution}/{name}', cache / name, args.offline)
    inception = Path(args.inception).expanduser().resolve() if args.inception else download(
        INCEPTION, cache / 'classify_image_graph_def.pb', args.offline)
    for path in (ref, inception):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    return ref, inception


def prepare_vae(args):
    if args.vae == 'mse':
        return download(
            IREPA + 'sdvae-ft-mse-f8d4.pt',
            Path(args.cache_dir).expanduser().resolve() / 'sdvae-ft-mse-f8d4.pt',
            args.offline,
            'db2202ed373c6a28e59f4ab1f9d5043774ac1e71a3cecb3f806cba2f5ca95d0a')
    from diffusers import AutoencoderKL
    target = Path(args.cache_dir).expanduser().resolve() / f'vae-{args.vae}'
    if (target / 'config.json').is_file():
        # Verify weights can be loaded, including offline mode.
        AutoencoderKL.from_pretrained(str(target), local_files_only=True)
    elif args.offline:
        raise FileNotFoundError(f'Missing VAE: {target}; run prepare online first')
    else:
        vae = AutoencoderKL.from_pretrained(f'stabilityai/sd-vae-ft-{args.vae}')
        vae.save_pretrained(str(target))
    return target


IREPA = 'https://huggingface.co/REPA-E/iREPA-collections/resolve/main/pretrained_models/'
STATS_SHA256 = '5e9e42b7d12764372c38644ad520a1ff254bed01fcb6b6c57f75917cee247449'


def prepare_stats(args):
    if args.latent_stats:
        path = Path(args.latent_stats).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    bundled = Path(__file__).resolve().parent / 'pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt'
    if bundled.is_file():
        if sha256(bundled) != STATS_SHA256:
            raise ValueError('Bundled latent stats failed official SHA256 check')
        return bundled
    return download(IREPA + 'sdvae-ft-mse-f8d4-latents-stats.pt',
                    Path(args.cache_dir).expanduser().resolve() / 'latent-stats.pt',
                    args.offline, STATS_SHA256)
