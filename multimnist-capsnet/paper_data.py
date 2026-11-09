"""Read-only, hash-bound original-order MNIST and deterministic streamed batches.

No download, extraction, cache mutation or ML-library import occurs here.
Train/test files are opened only through the explicitly selected split.
"""
import hashlib
import json
from pathlib import Path
from paper_protocol import PairStream, integer, render
from paper_state import ordinary

REVISION = "77f3279092a1c1579b2250db8eafed0ad422088c"
SPLITS = {
    "train": (60000, "f2c01285a9f89399335b00ee4e8d499dc4e46db5e39c74903ce5618d895eb3bf",
              "741c988805d008ac6e4c904b69001ba184c24b2c540a4ef403f4c71b676cf757",
              "1feba77c54802fa5339a11837ea4b2866434b83314ec45192930f1df69120c13"),
    "test": (10000, "d49fcf556ce25b002b302e318ce4a11098bbfe5d4499c3f35d7c72297c52374b",
             "6d87418db22cc8025d05968bec9bd5c3932904b23485740db143a061a2c9d161",
             "ddeff807876a9661a1110d45c266c86239a3a1b7d37da0c3716a7a683c852ff5"),
}


def bound_bytes(path, size, sha256):
    """Authenticate the exact returned buffer, never hash and then reopen it."""
    ordinary(path)
    if path.stat().st_size != size:
        raise ValueError("Bound input size differs")
    with path.open("rb") as handle:
        data = handle.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Bound input bytes differ")
    return data


def read_split(root, expected_manifest_sha256, split):
    """Requires a separately admitted manifest from complete parquet/IDX comparison.

    This is a dataset identity check, not experiment/test-release authority. The
    future runner must separately enforce the approved train/evaluation phase.
    """
    if split not in SPLITS:
        raise ValueError("Explicit train/test split required")
    root = Path(root).absolute()
    for path in reversed([root, *root.parents]):
        ordinary(path, directory=True)
    manifest_path = root / "MANIFEST.json"
    ordinary(manifest_path)
    size = manifest_path.stat().st_size
    if not 0 < size <= 16384:
        raise ValueError("Dataset metadata cap exceeded")
    manifest = json.loads(bound_bytes(manifest_path, size, expected_manifest_sha256))
    if manifest.get("schemaVersion") != 1 or manifest.get("providerRevision") != REVISION:
        raise ValueError("Pinned provider/manifest schema differs")
    count, parquet, image_sha, label_sha = SPLITS[split]
    row = manifest["splits"][split]
    if row["count"] != count or row["parquetSHA256"] != parquet:
        raise ValueError("Original complete split required")
    result = []
    for name, size, sha in (("images", count * 784, image_sha), ("labels", count, label_sha)):
        expected = {"path": split + "." + name, "size": size, "sha256": sha}
        if row[name] != expected:
            raise ValueError("Canonical original-order raw split identity differs")
        result.append(bound_bytes(root / expected["path"], size, sha))
    if any(label > 9 for label in result[1]):
        raise ValueError("Non-digit MNIST label")
    return tuple(result)


def batches(stream, images, batch_size, *, start=0, stop=None, exposure=0):
    """Return complete byte batches with next-cursor metadata, including tail rows."""
    if not isinstance(stream, PairStream) or len(images) != len(stream.labels) * 784:
        raise ValueError("Stream/source image cardinality mismatch")
    integer(batch_size, "batch size", 1, 128)
    integer(start, "start cursor", 0, stream.count)
    stop = stream.count if stop is None else integer(stop, "stop cursor", start, stream.count)
    cursor = start
    while cursor < stop:
        end = min(cursor + batch_size, stop)
        pairs = [stream.ordered(i, exposure) for i in range(cursor, end)]
        composites, first, second = [], [], []
        for pair in pairs:
            image_a = images[pair.anchor * 784:(pair.anchor + 1) * 784]
            image_b = images[pair.partner * 784:(pair.partner + 1) * 784]
            composite, targets = render(image_a, image_b, pair)
            composites.append(composite)
            first.append(targets[0])
            second.append(targets[1])
        yield {"startCursor": cursor, "nextCursor": end, "exposure": exposure,
               "pairs": pairs, "images": b"".join(composites),
               "targets": (b"".join(first), b"".join(second))}
        cursor = end
