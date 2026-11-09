import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_data import batches, bound_bytes, read_split
from paper_protocol import PairStream, Protocol, render


class DataTests(unittest.TestCase):
    def test_complete_tail_and_restart_match_individual_pair_reference(self):
        stream = PairStream(bytes(range(10)), "train", Protocol(pairs_per_anchor=3), fixture=True)
        images = b"".join(bytes([i * 20]) * 784 for i in range(10))
        result = list(batches(stream, images, 7))
        self.assertEqual([len(batch["pairs"]) for batch in result], [7, 7, 7, 7, 2])
        self.assertEqual([batch["nextCursor"] for batch in result], [7, 14, 21, 28, 30])
        self.assertEqual(list(batches(stream, images, 7, start=14)), result[2:])
        self.assertEqual({p.identity for batch in result for p in batch["pairs"]}, set(range(30)))
        for batch in result:
            for offset, pair in enumerate(batch["pairs"]):
                composite, targets = render(images[pair.anchor * 784:(pair.anchor + 1) * 784],
                                            images[pair.partner * 784:(pair.partner + 1) * 784], pair)
                cut = slice(offset * 1296, (offset + 1) * 1296)
                self.assertEqual(batch["images"][cut], composite)
                self.assertEqual(batch["targets"][0][cut], targets[0])
                self.assertEqual(batch["targets"][1][cut], targets[1])
        self.assertEqual(list(batches(stream, images, 7, start=30)), [])
        with self.assertRaises(ValueError):
            list(batches(stream, images[:-1], 7))

    def test_authenticated_buffers_and_manifest_fail_before_data_open(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            payload = root / "fixture.bin"
            payload.write_bytes(b"small fixture, not MNIST")
            sha = hashlib.sha256(payload.read_bytes()).hexdigest()
            self.assertEqual(bound_bytes(payload, payload.stat().st_size, sha), b"small fixture, not MNIST")
            with self.assertRaises(ValueError):
                bound_bytes(payload, payload.stat().st_size, "0" * 64)
            with self.assertRaises(ValueError):
                bound_bytes(payload, 1, sha)
            manifest = root / "MANIFEST.json"
            manifest.write_bytes(b'{"schemaVersion": 9}')
            with self.assertRaises(ValueError):
                read_split(root, hashlib.sha256(manifest.read_bytes()).hexdigest(), "train")
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["MANIFEST.json", "fixture.bin"])
            with self.assertRaises(ValueError):
                read_split(root, "0" * 64, "train")
            with self.assertRaises(ValueError):
                read_split(root, sha, "validation")


if __name__ == "__main__":
    unittest.main(verbosity=2)
