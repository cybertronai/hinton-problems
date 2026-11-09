import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_state import Checkpoints

BINDING = {name + "SHA256": str(i) * 64 for i, name in enumerate(("protocol", "source", "data", "runtime"))}


class CheckpointTests(unittest.TestCase):
    def test_resume_and_two_rolling_payloads(self):
        with tempfile.TemporaryDirectory() as root:
            store = Checkpoints(Path(root) / "checkpoints", BINDING, payload_limit=1024)
            self.assertIsNone(store.read())
            first = store.commit(b"model optimizer rng cursor=128", 128, 1, best=True)
            store.commit(b"model optimizer rng cursor=256", 256, 2)
            store.commit(b"model optimizer rng cursor=384", 384, 3)
            resumed = Checkpoints(store.root, BINDING, payload_limit=1024)
            self.assertEqual(resumed.read()[0], b"model optimizer rng cursor=384")
            self.assertEqual(resumed.read("best")[1], first)
            self.assertEqual(len(list(store.root.glob("payload-*.bin"))), 2)
            self.assertFalse((store.root / "WRITER.lock").exists())
            altered = dict(BINDING, sourceSHA256="f" * 64)
            with self.assertRaises(ValueError):
                Checkpoints(store.root, altered).read()

    def test_corruption_refused_before_deserialization(self):
        with tempfile.TemporaryDirectory() as root:
            store = Checkpoints(root, BINDING, payload_limit=64)
            entry = store.commit(b"trusted-state", 128, 1)
            (Path(root) / entry["name"]).write_bytes(b"changed-state")
            with self.assertRaises(ValueError):
                store.read()

    def test_interrupted_index_publication_preserves_old_pointer_and_lock(self):
        with tempfile.TemporaryDirectory() as root:
            store = Checkpoints(root, BINDING)
            first = store.commit(b"one", 128, 1)
            original = (Path(root) / "INDEX.json").read_bytes()
            with patch("paper_state.os.replace", side_effect=OSError("fixture interruption")):
                with self.assertRaises(OSError):
                    store.commit(b"two", 256, 2)
            self.assertEqual((Path(root) / "INDEX.json").read_bytes(), original)
            self.assertTrue((Path(root) / first["name"]).exists())
            self.assertTrue((Path(root) / "WRITER.lock").exists())
            with self.assertRaises(ValueError):
                store.read()
            with self.assertRaises(FileExistsError):
                store.commit(b"retry", 256, 2)

    def test_owned_paths_caps_and_nonadvancing_cursor(self):
        with tempfile.TemporaryDirectory() as root:
            store = Checkpoints(root, BINDING, payload_limit=8)
            with self.assertRaises(ValueError):
                store.commit(b"too-large", 128, 1)
            self.assertFalse((Path(root) / "WRITER.lock").exists())
            store.commit(b"one", 128, 1)
            with self.assertRaises(ValueError):
                store.commit(b"two", 128, 2)
            self.assertTrue((Path(root) / "WRITER.lock").exists())
        with tempfile.TemporaryDirectory() as root:
            store = Checkpoints(root, BINDING)
            store.commit(b"one", 128, 1)
            index = json.loads((Path(root) / "INDEX.json").read_bytes())
            index["latest"]["name"] = "../outside.bin"
            (Path(root) / "INDEX.json").write_text(json.dumps(index), encoding="utf-8")
            with self.assertRaises(ValueError):
                store.read()


if __name__ == "__main__":
    unittest.main(verbosity=2)
