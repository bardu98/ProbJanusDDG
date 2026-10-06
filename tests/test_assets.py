import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parent.parent / 'scripts/manage_assets.py'
spec = importlib.util.spec_from_file_location('asset_manager', SCRIPT)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


class AssetBundleTests(unittest.TestCase):
    def test_chunked_bundle_roundtrip_and_corruption_detection(self):
        previous_root, previous_chunk = assets.ROOT, assets.CHUNK
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                assets.ROOT, assets.CHUNK = root, 64
                (root / 'embeddings').mkdir()
                (root / 'pretrained/checkpoints').mkdir(parents=True)
                flat = root / 'embeddings/long_flat.npy'
                checkpoint = root / 'pretrained/checkpoints/model.pt'
                original = bytes(range(163))
                flat.write_bytes(original)
                checkpoint.write_bytes(b'checkpoint content')
                paths = [flat, checkpoint]
                manifest = {'files': {str(p.relative_to(root)): {
                    'bytes': p.stat().st_size, 'sha256': assets.digest(p),
                } for p in paths}}
                (root / 'ASSETS.json').write_text(json.dumps(manifest))
                folder = root / 'bundle'
                assets.pack(folder)
                self.assertEqual(len(list(folder.glob('*.part*'))), 3)
                flat.unlink(); checkpoint.unlink()
                assets.restore(folder)
                self.assertEqual(flat.read_bytes(), original)
                self.assertEqual(checkpoint.read_bytes(), b'checkpoint content')
                flat.unlink()
                part = folder / 'long_flat.npy.part001'
                part.write_bytes(b'x' * part.stat().st_size)
                with self.assertRaises(AssertionError):
                    assets.restore(folder)
                self.assertFalse(flat.exists())
        finally:
            assets.ROOT, assets.CHUNK = previous_root, previous_chunk


if __name__ == '__main__':
    unittest.main()
