import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from scripts.check_public_release import release_files


class ReceiptTests(unittest.TestCase):
    def make_release(self, root):
        path = root / "public_results/metrics_01.csv"
        path.parent.mkdir()
        path.write_text("model,cindex\nRF,0.7\n")
        manifest = {"schema": "cipher-public-release-v1", "sample_rows": False, "files": [{"file": "public_results/metrics_01.csv", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
        (root / "release_manifest.json").write_text(json.dumps(manifest))
        return path

    def test_matching_receipt_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_release(root)
            self.assertEqual(len(release_files(root)), 2)

    def test_unlisted_participant_table_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_release(root)
            (root / "sample_predictions.csv").write_text("eid,risk\n123,0.7\n")
            with self.assertRaises(ValueError):
                release_files(root)

    def test_modified_aggregate_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            table = self.make_release(root)
            table.write_text("eid,risk\n123,0.7\n")
            with self.assertRaises(ValueError):
                release_files(root)


if __name__ == "__main__":
    unittest.main()
