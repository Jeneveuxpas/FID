"""CPU regression checks; does not claim CUDA/checkpoint/FID runtime validation."""
import ast
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import assets
from fid import inspect_samples
import run

ROOT = Path(__file__).resolve().parent


class ToolTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict("os.environ", {"FID_DOWNLOAD_CONNECTIONS": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_download_cache_offline_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'asset'
            response = io.BytesIO(b'correct-asset')
            response.headers = {'Content-Length': '13'}
            with patch('urllib.request.urlopen', return_value=response):
                assets.download('https://example.test/asset', target)
            assets.download('https://example.test/asset', target, offline=True)
            target.write_bytes(b'corrupt-asset')
            with self.assertRaises(ValueError):
                assets.download('https://example.test/asset', target, offline=True)

    def test_partial_download_never_becomes_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'asset'
            def response(*args, **kwargs):
                result = io.BytesIO(b'partial')
                result.headers = {'Content-Length': '100'}
                return result
            with patch('urllib.request.urlopen', side_effect=response), patch('time.sleep'):
                with self.assertRaises(IOError):
                    assets.download('https://example.test/asset', target)
            self.assertFalse(target.exists())
            with self.assertRaises(FileNotFoundError):
                assets.download('https://example.test/asset', target, offline=True)

    def test_streaming_npz_matches_images_and_preserves_existing_on_failure(self):
        # Exercise the actual packer without importing CUDA dependencies.
        module = ast.parse((ROOT / 'generate.py').read_text())
        functions = [n for n in module.body if isinstance(n, ast.FunctionDef)
                     and n.name in ['valid_sample', 'create_npz_from_sample_folder']]
        namespace = {'np': np, 'Image': Image, 'os': os,
                     'tqdm': lambda iterable, **kwargs: iterable}
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'generate.py', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as folder:
            samples = Path(folder) / 'samples'
            samples.mkdir()
            images = np.random.default_rng(9).integers(0, 256, (3, 256, 256, 3), dtype=np.uint8)
            for i, image in enumerate(images):
                Image.fromarray(image).save(samples / f'{i:06d}.png')
            packed = namespace['create_npz_from_sample_folder'](samples, 3)
            self.assertEqual(inspect_samples(packed, 256, 3), 3)
            with np.load(packed) as data:
                np.testing.assert_array_equal(data['arr_0'], images)
            original = assets.sha256(packed)
            (samples / '000002.png').write_bytes(b'broken')
            with self.assertRaises(ValueError):
                namespace['create_npz_from_sample_folder'](samples, 3)
            self.assertEqual(assets.sha256(packed), original)
            with self.assertRaises(ValueError):
                inspect_samples(packed, 512)
            with self.assertRaises(ValueError):
                inspect_samples(packed, 256, 50000)

    def test_reject_float_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'samples.npz'
            np.savez(path, arr_0=np.zeros((2,256,256,3), dtype=np.float32))
            with self.assertRaises(ValueError):
                inspect_samples(path, 256)

    def test_sampler_arguments_and_no_implicit_resume(self):
        args = run.parser().parse_args(['run', '--ckpt', '/tmp/a b.pt', '--gpus', '0,2',
                                      '--model', 'SiT-XL/2', '--resolution', '256',
                                      '--cfg-scale', '1.7', '--guidance-high', '0.7', '--no-qk-norm'])
        args.path_type = 'linear'
        command = run.sampling_command(args, '/tmp/stats.pt', '/tmp/vae.pt', Path('/tmp/result'))
        self.assertIn('--nproc_per_node=2', command)
        self.assertIn('/tmp/a b.pt', command)
        self.assertIn('--no-qk-norm', command)
        self.assertNotIn('--resume-samples', command)
        self.assertEqual(command[command.index('--guidance-high') + 1], '0.7')

    def test_bundled_official_latent_stats_hash(self):
        args = run.parser().parse_args(['prepare', '--offline'])
        self.assertEqual(assets.sha256(assets.prepare_stats(args)), assets.STATS_SHA256)


if __name__ == '__main__':
    unittest.main()
