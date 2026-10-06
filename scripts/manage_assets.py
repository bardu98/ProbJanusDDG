"""Verify archived assets, package upload parts, or restore downloaded parts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent
CHUNK = 1024 ** 3


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(quick=False):
    manifest = json.loads((ROOT / 'ASSETS.json').read_text())
    failures = []
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file():
            failures.append(name + ': missing')
        elif path.stat().st_size != expected['bytes']:
            failures.append(name + ': incorrect size')
        elif not quick and digest(path) != expected['sha256']:
            failures.append(name + ': checksum mismatch')
    if failures:
        raise RuntimeError('\n'.join(failures) + '\nRestore the separate asset bundle as described in README.md.')
    print('Verified', len(manifest['files']), 'assets', '(sizes only)' if quick else '(SHA-256)')


def pack(folder):
    verify()
    folder.mkdir(parents=True, exist_ok=True)
    parts = []
    with (ROOT / 'embeddings/long_flat.npy').open('rb') as src:
        index = 0
        while True:
            block = src.read(min(8 * 1024 * 1024, CHUNK))
            if not block:
                break
            path = folder / f'long_flat.npy.part{index:03d}'
            count = 0
            with path.open('wb') as target:
                while block:
                    target.write(block)
                    count += len(block)
                    if count == CHUNK:
                        break
                    block = src.read(min(8 * 1024 * 1024, CHUNK - count))
            parts.append({'file': path.name, 'bytes': count, 'sha256': digest(path)})
            index += 1
    checkpoint_zip = folder / 'checkpoints.zip'
    with zipfile.ZipFile(checkpoint_zip, 'w', compression=zipfile.ZIP_STORED) as z:
        for path in sorted((ROOT / 'pretrained/checkpoints').glob('*.pt')):
            z.write(path, arcname=str(path.relative_to(ROOT)))
    metadata = {'parts': parts, 'checkpoints_zip_sha256': digest(checkpoint_zip),
                'flat_sha256': digest(ROOT / 'embeddings/long_flat.npy')}
    (folder / 'bundle.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Upload all files from', folder, 'alongside the source repository.')


def restore(folder):
    bundle = json.loads((folder / 'bundle.json').read_text())
    expected = json.loads((ROOT / 'ASSETS.json').read_text())['files']
    assert bundle['flat_sha256'] == expected['embeddings/long_flat.npy']['sha256']
    flat = ROOT / 'embeddings/long_flat.npy'
    if not flat.exists():
        flat.parent.mkdir(exist_ok=True)
        tmp = flat.with_suffix('.partial')
        with tmp.open('wb') as out:
            for part in bundle['parts']:
                if Path(part['file']).name != part['file']:
                    raise ValueError('Invalid asset filename')
                path = folder / part['file']
                assert path.stat().st_size == part['bytes'] and digest(path) == part['sha256']
                with path.open('rb') as src:
                    shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
        assert digest(tmp) == expected['embeddings/long_flat.npy']['sha256']
        tmp.replace(flat)
    archive = folder / 'checkpoints.zip'
    assert digest(archive) == bundle['checkpoints_zip_sha256']
    checkpoint_names = {name for name in expected if name.startswith('pretrained/checkpoints/')}
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == checkpoint_names
        for name in sorted(checkpoint_names):
            path = ROOT / name
            if path.exists():
                assert digest(path) == expected[name]['sha256']
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.partial')
            with z.open(name) as src, tmp.open('wb') as out:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
            assert tmp.stat().st_size == expected[name]['bytes'] and digest(tmp) == expected[name]['sha256']
            tmp.replace(path)
    verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('verify').add_argument('--quick', action='store_true')
    sub.add_parser('pack').add_argument('--output', type=Path, default=ROOT / 'dist/assets')
    sub.add_parser('restore').add_argument('folder', type=Path)
    args = parser.parse_args()
    if args.action == 'verify':
        verify(args.quick)
    elif args.action == 'pack':
        pack(args.output.resolve())
    else:
        restore(args.folder.resolve())


if __name__ == '__main__':
    main()
