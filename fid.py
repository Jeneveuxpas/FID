"""FID-only entry point using the unchanged SIT/ADM feature graph and distance."""
import argparse
import json
from pathlib import Path
import zipfile


def inspect_samples(path, resolution, expected_count=None):
    import numpy as np
    with zipfile.ZipFile(path) as archive, archive.open('arr_0.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f'Unsupported NPY version: {version}')
    if len(shape) != 4 or shape[1:] != (resolution, resolution, 3) or shape[0] < 2:
        raise ValueError(f'Expected [N>=2,{resolution},{resolution},3], got {shape}')
    if dtype != np.uint8 or fortran:
        raise ValueError('Samples must be C-contiguous uint8 RGB images in [0,255]')
    if expected_count is not None and shape[0] != expected_count:
        raise ValueError(f'Expected {expected_count} samples; found {shape[0]}')
    return shape[0]


def evaluate(args):
    import numpy as np
    count = inspect_samples(args.samples, args.resolution, args.expected_count)
    import adm_evaluator as adm
    adm.tf.disable_eager_execution()
    adm.INCEPTION_V3_PATH = str(Path(args.inception).resolve())
    with np.load(args.ref_batch, allow_pickle=False) as reference:
        mu, sigma = reference['mu'], reference['sigma']
        if mu.shape != (2048,) or sigma.shape != (2048, 2048):
            raise ValueError('Reference NPZ must contain ADM mu[2048] and sigma[2048,2048]')
        ref_stats = adm.FIDStatistics(mu, sigma)
    config = adm.tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with adm.tf.Graph().as_default(), adm.tf.Session(config=config) as session:
        inputs = adm.tf.placeholder(adm.tf.float32, shape=[None, None, None, 3])
        pool, _ = adm._create_feature_graph(inputs)
        features = []
        with adm.open_npz_array(args.samples, 'arr_0') as reader:
            for batch in adm.tqdm(reader.read_batches(args.batch_size), desc='FID features'):
                values = session.run(pool, {inputs: batch.astype(np.float32)})
                features.append(values.reshape(len(batch), -1))
        features = np.concatenate(features, axis=0)
        stats = adm.FIDStatistics(np.mean(features, axis=0), np.cov(features, rowvar=False))
        value = float(stats.frechet_distance(ref_stats))
    if not np.isfinite(value):
        raise ValueError(f'Non-finite FID: {value}')
    result = {'FID': value, 'num_samples': count, 'resolution': args.resolution,
              'sample_npz': str(Path(args.samples).resolve()),
              'reference_npz': str(Path(args.ref_batch).resolve()),
              'inception': str(Path(args.inception).resolve()),
              'implementation': 'SIT OpenAI ADM TensorFlow pool_3',
              'numpy': np.__version__, 'tensorflow': adm.tf.__version__}
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.partial')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(target)
    print(f'FID: {value:.8f}\nResults: {target}', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples', required=True)
    p.add_argument('--ref-batch', required=True)
    p.add_argument('--inception', required=True)
    p.add_argument('--resolution', type=int, choices=[256,512], default=256)
    p.add_argument('--expected-count', type=int)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    evaluate(args)
