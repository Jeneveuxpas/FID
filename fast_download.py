"""Parallel HTTP ranges with persistent chunks and representation validation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time
import urllib.request


def parallel_download(url, destination, connections=16):
    """Return False when range downloads cannot safely be used."""
    if connections <= 1:
        return False
    try:
        request = urllib.request.Request(url, method='HEAD', headers={'Accept-Encoding': 'identity'})
        with urllib.request.urlopen(request, timeout=30) as response:
            size = int(response.headers.get('Content-Length', 0))
            etag = response.headers.get('ETag', '')
            ranges = response.headers.get('Accept-Ranges', '').lower()
    except (OSError, ValueError):
        return False
    # A strong ETag prevents mixing chunks from different object versions.
    if size < 8 * 1024 * 1024 or not etag or etag.startswith('W/'):
        return False
    # Azure may omit Accept-Ranges on HEAD; verify actual range behavior.
    probe = urllib.request.Request(url, headers={
        'Range': 'bytes=0-0', 'If-Range': etag, 'Accept-Encoding': 'identity'})
    with urllib.request.urlopen(probe, timeout=30) as response:
        if (response.status != 206 or
                response.headers.get('Content-Range') != f'bytes 0-0/{size}' or
                response.headers.get('ETag') != etag):
            return False
    connections = min(connections, 32)
    chunk_size = max(4 * 1024 * 1024, (size + connections - 1) // connections)
    directory = Path(str(destination) + '.chunks')
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'manifest.json'
    identity = {'url': url, 'size': size, 'etag': etag, 'chunk_size': chunk_size}
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        # Only remove our numbered download chunks, never unrelated user files.
        for part in directory.glob('*.chunk'):
            part.unlink()
    manifest.write_text(json.dumps(identity))
    started = time.monotonic()

    def fetch(index, start, end):
        part = directory / f'{index:05d}.chunk'
        wanted = end - start + 1
        for attempt in range(3):
            have = part.stat().st_size if part.exists() else 0
            if have == wanted:
                return wanted
            if have > wanted:
                part.unlink()
                have = 0
            try:
                request = urllib.request.Request(url, headers={
                    'Range': f'bytes={start + have}-{end}', 'If-Range': etag,
                    'Accept-Encoding': 'identity'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected_range = f'bytes {start + have}-{end}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                        raise ValueError('Server ignored range or asset changed; refusing mixed download')
                    if response.headers.get('ETag') != etag:
                        raise ValueError('Asset ETag changed during download')
                    with part.open('ab') as output:
                        remaining = wanted - have
                        while remaining:
                            data = response.read(min(1024 * 1024, remaining))
                            if not data:
                                raise IOError('Incomplete range response')
                            output.write(data)
                            remaining -= len(data)
                return wanted
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    jobs = [(i, start, min(start + chunk_size, size) - 1)
            for i, start in enumerate(range(0, size, chunk_size))]
    print(f'Parallel download: {connections} connections, {size / 2**20:.1f} MiB; resumable chunks', flush=True)
    completed = 0
    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [pool.submit(fetch, *job) for job in jobs]
        for future in as_completed(futures):
            completed += future.result()
            print(f'  {completed / 2**20:.1f}/{size / 2**20:.1f} MiB complete '
                  f'({time.monotonic() - started:.0f}s)', flush=True)
    with Path(destination).open('wb') as output:
        for index, _, _ in jobs:
            with (directory / f'{index:05d}.chunk').open('rb') as source:
                while True:
                    block = source.read(4 * 1024 * 1024)
                    if not block:
                        break
                    output.write(block)
    return True


def clear_chunks(destination):
    directory = Path(str(destination) + '.chunks')
    if directory.exists():
        for part in directory.glob('*.chunk'):
            part.unlink()
        (directory / 'manifest.json').unlink(missing_ok=True)
        directory.rmdir()
