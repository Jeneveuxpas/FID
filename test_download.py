import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from fast_download import parallel_download


class ParallelTests(unittest.TestCase):
    def test_ranges_resume_and_assembly(self):
        data = bytes(range(256)) * (9 * 4096)
        etag = '"version1"'
        calls = []
        def serve(request, **kwargs):
            if request.get_method() == 'HEAD':
                response = io.BytesIO()
                response.headers = {'Content-Length': str(len(data)), 'ETag': etag, 'Accept-Ranges': 'bytes'}
                return response
            value = request.get_header('Range')
            calls.append(value)
            start, end = map(int, value[6:].split('-'))
            response = io.BytesIO(data[start:end+1])
            response.status = 206
            response.headers = {'Content-Range': f'bytes {start}-{end}/{len(data)}', 'ETag': etag}
            return response
        with tempfile.TemporaryDirectory() as folder, patch('urllib.request.urlopen', side_effect=serve):
            dest = Path(folder) / 'asset.partial'
            self.assertTrue(parallel_download('https://example.test/file', dest, 3))
            self.assertEqual(dest.read_bytes(), data)
            calls.clear()
            # Simulate a partially downloaded chunk before a restarted transfer.
            part = Path(str(dest)+'.chunks')/'00000.chunk'
            part.write_bytes(part.read_bytes()[:100])
            self.assertTrue(parallel_download('https://example.test/file', dest, 3))
            self.assertEqual(dest.read_bytes(), data)
            self.assertEqual(calls, ['bytes=0-0', 'bytes=100-4194303'])

    def test_ignored_range_is_rejected(self):
        def serve(request, **kwargs):
            response = io.BytesIO()
            response.headers = {'Content-Length': str(9*1024*1024), 'ETag': '"v1"', 'Accept-Ranges': 'bytes'}
            response.status = 200
            return response
        with tempfile.TemporaryDirectory() as folder, patch('urllib.request.urlopen', side_effect=serve):
            destination = Path(folder)/'asset.partial'
            self.assertFalse(parallel_download('https://example.test/file', destination, 2))
            self.assertFalse(destination.exists())
