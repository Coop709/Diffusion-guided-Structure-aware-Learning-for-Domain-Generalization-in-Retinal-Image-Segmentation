import json
import tempfile
import unittest
from pathlib import Path

from datasets.manifest import read_manifest


class ManifestTests(unittest.TestCase):
    def test_labeled_vessel_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "image.png").touch()
            (root / "mask.png").touch()
            manifest = root / "source.jsonl"
            manifest.write_text(json.dumps({"image": "image.png", "mask": "mask.png", "domain": "DRIVE"}),
                                encoding="utf-8")
            records = read_manifest(str(manifest), "vessel", require_labels=True)
            self.assertEqual(records[0]["domain"], "DRIVE")

    def test_missing_mask_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "image.png").touch()
            manifest = root / "source.jsonl"
            manifest.write_text('{"image":"image.png"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mask"):
                read_manifest(str(manifest), "vessel", require_labels=True)

    def test_target_does_not_require_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "image.png").touch()
            manifest = root / "target.jsonl"
            manifest.write_text('{"image":"image.png"}', encoding="utf-8")
            self.assertEqual(len(read_manifest(str(manifest), "odoc", require_labels=False)), 1)


if __name__ == "__main__":
    unittest.main()
