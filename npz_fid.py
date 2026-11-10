#!/usr/bin/env python3
"""Download ADM assets and evaluate an existing sample NPZ in the current environment."""
import argparse
import importlib.util
import os
from pathlib import Path
from assets import prepare_metric_assets
from fid import inspect_samples, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('samples', nargs='?', help='Generated sample NPZ; not a model checkpoint')
    p.add_argument('--prepare', action='store_true', help='Only download reference NPZ and Inception')
    p.add_argument('--resolution', type=int, choices=[256, 512], default=256)
    p.add_argument('--cache-dir', default=str(Path(__file__).resolve().parent / 'assets'))
    p.add_argument('--ref-batch', help='Existing matching ADM reference NPZ')
    p.add_argument('--inception', help='Existing classify_image_graph_def.pb')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--gpu', default='0', help='One GPU ID, or cpu; FID is single-process')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--expected-count', type=int, help='Optional strict sample count, e.g. 50000')
    p.add_argument('--output', help='Default: <samples>.fid.json')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('--batch-size must be positive')
    if args.gpu != 'cpu' and not args.gpu.isdigit():
        p.error('--gpu must be one integer GPU ID or cpu')
    if args.expected_count is not None and args.expected_count < 2:
        p.error('--expected-count must be >= 2')
    if not args.prepare:
        if not args.samples:
            p.error('Provide a sample NPZ, or use --prepare')
        missing = [name for name in ['numpy', 'scipy', 'tensorflow', 'requests', 'tqdm']
                   if importlib.util.find_spec(name) is None]
        if missing:
            p.error('Missing FID dependencies: ' + ', '.join(missing) +
                    '. See README; the training environment may not include TensorFlow.')
        args.samples = str(Path(args.samples).expanduser().resolve())
        inspect_samples(args.samples, args.resolution, args.expected_count)
        args.output = str(Path(args.output).expanduser().resolve()) if args.output else args.samples + '.fid.json'
        if Path(args.output) == Path(args.samples):
            p.error('--output must not overwrite the sample NPZ')
    ref, inception = prepare_metric_assets(args)
    if args.prepare:
        print(f'Ready:\n  {ref}\n  {inception}')
        return
    args.ref_batch, args.inception = str(ref), str(inception)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' if args.gpu == 'cpu' else args.gpu
    evaluate(args)


if __name__ == '__main__':
    main()
