"""Small transactional local checkpoint store, independent of tensor libraries.

Payload serialization belongs to the future trainer. The store binds the exact
serialized bytes before a caller may deserialize them. Never load an untrusted
pickle merely because this module can compute its hash.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import uuid
from paper_protocol import canonical, integer


def ordinary(path, directory=False):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Linked/reparse checkpoint path")
    if not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError("Wrong checkpoint path type")


class Checkpoints:
    """Latest/best pointers, atomic readback, fail-closed interrupted commits.

    An interrupted writer leaves its exclusive lock for manual reconciliation.
    No automatic stale-lock stealing or arbitrary-directory cleanup exists.
    """
    def __init__(self, root, binding, *, payload_limit=256 * 1024**2):
        self.root = Path(root).absolute()
        for path in reversed([self.root, *self.root.parents]):
            if path.exists() or path.is_symlink():
                ordinary(path, directory=True)
        self.root.mkdir(exist_ok=True)
        if set(binding) != {"protocolSHA256", "sourceSHA256", "dataSHA256", "runtimeSHA256"}:
            raise ValueError("Incomplete checkpoint identity")
        if any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v) for v in binding.values()):
            raise ValueError("Malformed checkpoint identity")
        self.binding = dict(binding)
        self.limit = integer(payload_limit, "payload limit", 1, 256 * 1024**2)

    @contextmanager
    def _writer(self):
        lock = self.root / "WRITER.lock"
        with lock.open("xb") as handle:
            handle.write(canonical({"binding": self.binding, "pid": os.getpid()}))
            handle.flush()
            os.fsync(handle.fileno())
        completed = False
        try:
            yield
            completed = True
        finally:
            if completed:
                ordinary(lock)
                lock.unlink()

    def _index(self):
        path = self.root / "INDEX.json"
        if not path.exists() and not path.is_symlink():
            if any(self.root.glob("payload-*.bin")):
                raise ValueError("Unreferenced checkpoint payload; manual reconciliation required")
            return {"schemaVersion": 1, "binding": self.binding, "latest": None, "best": None}
        ordinary(path)
        if path.stat().st_size > 16384:
            raise ValueError("Checkpoint index cap exceeded")
        result = json.loads(path.read_bytes())
        if set(result) != {"schemaVersion", "binding", "latest", "best"} or result["schemaVersion"] != 1 or result["binding"] != self.binding:
            raise ValueError("Checkpoint source/data/runtime/protocol identity mismatch")
        return result

    def _payload(self, entry):
        if not isinstance(entry, dict) or set(entry) != {"name", "sha256", "size", "completedPairs", "completedUpdates"}:
            raise ValueError("Malformed checkpoint entry")
        if not re.fullmatch(r"payload-[0-9a-f]{32}\.bin", entry["name"]):
            raise ValueError("Non-owned payload path")
        path = self.root / entry["name"]
        ordinary(path)
        size = integer(entry["size"], "payload bytes", 1, self.limit)
        if path.stat().st_size != size:
            raise ValueError("Checkpoint length changed")
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Checkpoint payload changed")
        integer(entry["completedPairs"], "completed pairs", 0, 60000000 * 100)
        integer(entry["completedUpdates"], "completed updates", 0, 60000000 * 100)
        return data

    def read(self, which="latest"):
        if which not in ("latest", "best"):
            raise ValueError("Unsupported checkpoint pointer")
        if (self.root / "WRITER.lock").exists() or (self.root / "WRITER.lock").is_symlink():
            raise ValueError("Active/interrupted checkpoint writer; reconcile before reading")
        entry = self._index()[which]
        if entry is None:
            return None
        return self._payload(entry), dict(entry)

    def commit(self, payload, completed_pairs, completed_updates, *, best=False):
        if type(payload) is not bytes or not 0 < len(payload) <= self.limit or type(best) is not bool:
            raise ValueError("Invalid bounded checkpoint payload")
        integer(completed_pairs, "completed pairs", 0, 60000000 * 100)
        integer(completed_updates, "completed updates", 0, 60000000 * 100)
        with self._writer():
            index = self._index()
            old = {e["name"]: e for e in (index["latest"], index["best"]) if e is not None}
            for entry in old.values():
                self._payload(entry)  # all old pointers must remain authentic
            if index["latest"] is not None and (completed_pairs <= index["latest"]["completedPairs"] or
                                                completed_updates <= index["latest"]["completedUpdates"]):
                raise ValueError("Checkpoint cursor did not advance")
            # At most two committed payloads plus one transient payload. Unknown
            # leftovers are evidence, never silently retired or overwritten.
            if {p.name for p in self.root.glob("payload-*.bin")} != set(old):
                raise ValueError("Unexpected payload; manual reconciliation required")
            name = "payload-" + uuid.uuid4().hex + ".bin"
            entry = {"name": name, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
                     "completedPairs": completed_pairs, "completedUpdates": completed_updates}
            with (self.root / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if self._payload(entry) != payload:
                raise ValueError("Checkpoint readback failed")
            index["latest"] = entry
            if best:
                index["best"] = entry
            temporary = self.root / ("index-" + uuid.uuid4().hex + ".tmp")
            raw = canonical(index) + b"\n"
            with temporary.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            if temporary.read_bytes() != raw:
                raise ValueError("Checkpoint index readback failed")
            os.replace(temporary, self.root / "INDEX.json")
            if self._index() != index:
                raise ValueError("Published checkpoint readback failed")
            retained = {e["name"] for e in (index["latest"], index["best"]) if e is not None}
            for name, previous in old.items():
                if name not in retained:
                    self._payload(previous)  # only verified exact owned superseded bytes
                    (self.root / name).unlink()
        return entry
