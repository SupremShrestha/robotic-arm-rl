import hashlib
import json
from pathlib import Path
import unittest


class ModelMetadataTests(unittest.TestCase):
    def test_selected_checkpoint_hashes_match_metadata(self):
        root=Path(__file__).resolve().parents[1]
        checked=0
        for path in (root/"models").glob("*selection.json"):
            data=json.loads(path.read_text())
            if "sha256" not in data:continue
            with self.subTest(metadata=path.name):
                checkpoint=root/data["checkpoint"]
                self.assertTrue(checkpoint.is_file())
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),data["sha256"])
            checked+=1
        self.assertGreaterEqual(checked,10)
