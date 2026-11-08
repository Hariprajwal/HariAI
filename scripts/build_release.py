"""Validate the local model bundle before building a wheel and source archive."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / 'hari_ai' / 'assets'
    manifest = json.loads((assets / 'manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = assets / item['name']
        if path.parent.resolve() != assets.resolve():
            raise ValueError('Manifest entries must be asset basenames.')
        if path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError(f'Model bundle mismatch: {path.name}')
    subprocess.run([sys.executable, '-m', 'build', '--no-isolation'], cwd=root, check=True)
    import re
    version = re.search(r'^version\s*=\s*"([^"]+)"', (root/'pyproject.toml').read_text(encoding='utf-8'), re.M)[1]
    wheel = root / 'dist' / f'hari_ai-{version}-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as archive:
        for item in manifest['files']:
            data = archive.read('hari_ai/assets/' + item['name'])
            if hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError(f'Wheel checksum mismatch: {item["name"]}')
        for name in ['hari_ai/assets/LICENSE-SMOLLM', 'hari_ai/assets/MODEL_CARD.md',
                     'hari_ai/assets/manifest.json', 'hari_ai/vendor/open_jev/LICENSE']:
            archive.getinfo(name)
    print(f'Verified model-bearing wheel: {wheel} ({wheel.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
