import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import assets
from fid import inspect_samples

class ToolTests(unittest.TestCase):

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

    def test_reject_float_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'samples.npz'
            np.savez(path, arr_0=np.zeros((2, 256, 256, 3), dtype=np.float32))
            with self.assertRaises(ValueError):
                inspect_samples(path, 256)

if __name__ == "__main__":
    unittest.main()
