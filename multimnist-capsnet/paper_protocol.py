"""Independent, standard-library MultiMNIST protocol; no downloads or ML imports.

The released-author and paper-equation losses are deliberately separate. See
PAPER_PROTOCOL.md before interpreting either condition as a reproduction.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


@dataclass(frozen=True)
class Protocol:
    seed: int = 20261107
    pairs_per_anchor: int = 1000
    profile: str = "author2017"
    batch_size: int = 128
    exposures: int = 1
    validation_anchors: int = 1000
    validation_every_updates: int = 10000
    # An explicit proposal, not a recovered paper training duration.
    selection: str = "final-checkpoint-only"

    def __post_init__(self):
        integer(self.seed, "seed", 0, 2**32 - 1)
        integer(self.pairs_per_anchor, "pairs_per_anchor", 1, 1000)
        integer(self.batch_size, "batch_size", 1, 128)
        integer(self.exposures, "exposures", 1, 100)
        integer(self.validation_anchors, "validation anchors", 1, 60000)
        integer(self.validation_every_updates, "validation cadence", 1, 60000000)
        if self.profile not in ("author2017", "paper-equations"):
            raise ValueError("Unknown scientific condition")
        if self.selection != "final-checkpoint-only":
            raise ValueError("Test-based or unspecified checkpoint selection is not supported")

    @property
    def sha256(self):
        return digest(asdict(self))

    def describe(self):
        return {
            "protocol": asdict(self), "protocolSHA256": self.sha256,
            "trainPairs": 60000 * self.pairs_per_anchor,
            "testPairs": 10000 * self.pairs_per_anchor,
            "updatesPerExposure": math.ceil(60000 * self.pairs_per_anchor / self.batch_size),
            "parameters": 11361968 if self.profile == "author2017" else 11361808,
            "inputShape": [1, 36, 36], "primaryShape": [3200, 8],
            "digitShape": [10, 16], "routingIterations": 3,
            "decoder": [160, 512, 1024, 1296],
            "executionApproved": False, "measuredThroughput": None,
        }


class Permutation:
    """Six-round balanced Feistel PRP, restricted by bounded cycle walking.

    This defines our reproducible ordering, not the original NumPy RNG stream.
    No array proportional to the domain size is constructed.
    """

    def __init__(self, count, key):
        self.count = integer(count, "count", 1, 2**62)
        self.half = max(1, ((count - 1).bit_length() + 1) >> 1)
        self.mask = (1 << self.half) - 1
        self.key = hashlib.sha256(canonical(key)).digest()

    def _round(self, value, number):
        data = value.to_bytes(8, "little") + bytes([number])
        return int.from_bytes(hashlib.blake2s(data, key=self.key, digest_size=8).digest(), "little") & self.mask

    def _raw(self, value, inverse):
        left, right = value >> self.half, value & self.mask
        if inverse:
            for number in reversed(range(6)):
                left, right = right ^ self._round(left, number), left
        else:
            for number in range(6):
                left, right = right, left ^ self._round(right, number)
        return left << self.half | right

    def at(self, index, inverse=False):
        integer(index, "index", 0, self.count - 1)
        for _ in range(1024):
            index = self._raw(index, inverse)
            if index < self.count:
                return index
        raise RuntimeError("Permutation cycle-walk bound exceeded; no sample was substituted")


def choice(count, key):
    """Unbiased deterministic hash/rejection choice in a small finite domain."""
    integer(count, "choice count", 1, 2**32)
    cutoff = 2**64 - 2**64 % count
    for attempt in range(1024):
        raw = int.from_bytes(hashlib.blake2s(canonical([key, attempt]), digest_size=8).digest(), "little")
        if raw < cutoff:
            return raw % count
    raise RuntimeError("Hash choice bound exceeded")


@dataclass(frozen=True)
class Pair:
    split: str
    identity: int
    anchor: int
    partner: int
    labels: tuple
    anchor_shift: tuple
    partner_shift: tuple


class PairStream:
    """Index-only generator, same-split/different-label, unique partners/anchor.

    Pass the complete admitted labels for production. Synthetic small domains
    are explicit fixtures and must never be reported as MNIST measurements.
    """

    def __init__(self, labels, split, protocol, *, fixture=False):
        if split not in ("train", "test") or type(fixture) is not bool:
            raise ValueError("Expected explicit train/test split")
        self.labels = bytes(labels)
        if not self.labels or any(label > 9 for label in self.labels):
            raise ValueError("Labels must be nonempty integer digits 0..9")
        if not fixture and len(self.labels) != {"train": 60000, "test": 10000}[split]:
            raise ValueError("Production split cardinality mismatch")
        self.split, self.protocol = split, protocol
        self.eligible = {label: tuple(i for i, other in enumerate(self.labels) if label != other)
                         for label in set(self.labels)}
        if any(len(rows) < protocol.pairs_per_anchor for rows in self.eligible.values()):
            raise ValueError("Not enough different-label partners for unique pairing")
        self.count = len(self.labels) * protocol.pairs_per_anchor

    def pair(self, identity):
        integer(identity, "pair identity", 0, self.count - 1)
        anchor, counter = divmod(identity, self.protocol.pairs_per_anchor)
        return self._pair(anchor, counter, identity, self.split)

    def validation(self, anchor):
        """One extra ordered partner/anchor, not a source-image holdout.

        The train/test streams remain separate. This optional train-only
        diagnostic must not select the checkpoint used for official test.
        """
        if self.split != "train":
            raise ValueError("Validation must not consume test-source images")
        integer(anchor, "validation anchor", 0, len(self.labels) - 1)
        rows = self.eligible[self.labels[anchor]]
        if len(rows) <= self.protocol.pairs_per_anchor:
            raise ValueError("No unused validation partner remains")
        return self._pair(anchor, self.protocol.pairs_per_anchor, self.count + anchor, "train-diagnostic")

    def _pair(self, anchor, counter, identity, split):
        base = [self.protocol.seed, self.split, anchor]
        rows = self.eligible[self.labels[anchor]]
        partner = rows[Permutation(len(rows), [base, "partners"]).at(counter)]
        a = choice(81, [base, "anchor-shift"])
        b = choice(81, [base, counter, "partner-shift"])
        return Pair(split, identity, anchor, partner,
                    (self.labels[anchor], self.labels[partner]),
                    (a // 9 - 4, a % 9 - 4), (b // 9 - 4, b % 9 - 4))

    def ordered(self, cursor, exposure=0):
        integer(exposure, "exposure", 0, self.protocol.exposures - 1)
        order = Permutation(self.count, [self.protocol.seed, self.split, exposure, "order"])
        return self.pair(order.at(cursor))


def render(image_a, image_b, pair):
    """Byte-exact reference targets and clipped-addition composite (36x36)."""
    if len(image_a) != 784 or len(image_b) != 784:
        raise ValueError("Expected two raw uint8 28x28 images")
    targets = []
    for image, shift in ((image_a, pair.anchor_shift), (image_b, pair.partner_shift)):
        if len(shift) != 2 or any(type(s) is not int or not -4 <= s <= 4 for s in shift):
            raise ValueError("Shift outside the paper's signed four-pixel envelope")
        canvas = bytearray(1296)
        y, x = 4 + shift[0], 4 + shift[1]
        for row in range(28):
            canvas[(y + row) * 36 + x:(y + row) * 36 + x + 28] = image[row * 28:row * 28 + 28]
        targets.append(bytes(canvas))
    return bytes(min(a + b, 255) for a, b in zip(*targets)), tuple(targets)


def overlap_stats(targets):
    """Diagnostics at nonzero uint8 ink support; never an admission predicate."""
    if len(targets) != 2 or any(len(row) != 1296 for row in targets):
        raise ValueError("Expected two 36x36 byte targets")
    support = [{i for i, value in enumerate(row) if value > 0} for row in targets]
    intersection, union = len(support[0] & support[1]), len(support[0] | support[1])
    boxes = []
    for points in support:
        if not points:
            boxes.append(set())
        else:
            ys, xs = [i // 36 for i in points], [i % 36 for i in points]
            boxes.append({y * 36 + x for y in range(min(ys), max(ys) + 1)
                          for x in range(min(xs), max(xs) + 1)})
    box_union = len(boxes[0] | boxes[1])
    return {"inkPixels": [len(row) for row in support], "inkIntersection": intersection,
            "inkUnion": union, "inkIoU": intersection / union if union else None,
            "bboxIoU": len(boxes[0] & boxes[1]) / box_union if box_union else None,
            "admissionThreshold": None}


def squash(vector):
    if not vector or not all(math.isfinite(x) for x in vector):
        raise ValueError("Non-finite or empty capsule")
    norm = math.hypot(*vector)
    # Stable at zero and for large finite vector components.
    factor = (norm / (1 + norm * norm)) if norm <= 1 else (1 / norm) / (1 + (1 / norm)**2)
    return [x * factor for x in vector]


def route(votes, iterations=3, bias=None):
    """Tiny scalar reference: votes[input capsule][output class][atom]."""
    integer(iterations, "routing iterations", 1, 10)
    if not votes or not votes[0] or not votes[0][0]:
        raise ValueError("Empty votes")
    n, j, d = len(votes), len(votes[0]), len(votes[0][0])
    if any(len(v) != j or any(len(x) != d for x in v) for v in votes):
        raise ValueError("Ragged votes")
    if not all(math.isfinite(x) for v in votes for row in v for x in row):
        raise ValueError("Non-finite vote")
    bias = [[0.] * d for _ in range(j)] if bias is None else bias
    if len(bias) != j or any(len(row) != d for row in bias):
        raise ValueError("Ragged routing bias")
    logits = [[0.] * j for _ in range(n)]
    for iteration in range(iterations):
        coefficients = []
        for row in logits:
            exp = [math.exp(x - max(row)) for x in row]
            coefficients.append([x / math.fsum(exp) for x in exp])
        out = [squash([bias[k][a] + math.fsum(coefficients[i][k] * votes[i][k][a] for i in range(n))
                       for a in range(d)]) for k in range(j)]
        if iteration + 1 < iterations:
            logits = [[logits[i][k] + math.fsum(x * y for x, y in zip(votes[i][k], out[k]))
                       for k in range(j)] for i in range(n)]
    return out, coefficients


def margin(scores, labels, profile):
    if len(scores) != 10 or len(labels) != 2 or len(set(labels)) != 2 or any(type(x) is not int or not 0 <= x < 10 for x in labels):
        raise ValueError("Expected 10 scores and two distinct digit labels")
    if not all(math.isfinite(x) and x >= 0 for x in scores):
        raise ValueError("Capsule lengths must be finite/nonnegative")
    value = math.fsum(max(.9 - x, 0)**2 if k in labels else .5 * max(x - .1, 0)**2
                      for k, x in enumerate(scores))
    if profile == "author2017":
        return value / 20  # released code: .5 * mean over 10 classes
    if profile == "paper-equations":
        return value
    raise ValueError("Unknown loss profile")


def reconstruction_loss(predicted, targets):
    if len(predicted) != 2 or len(targets) != 2 or any(len(row) != 1296 for row in (*predicted, *targets)):
        raise ValueError("Expected two paired 36x36 normalized reconstructions")
    if any(not math.isfinite(x) or not 0 <= x <= 1 for row in (*predicted, *targets) for x in row):
        raise ValueError("Invalid normalized reconstruction pixels")
    return .0005 * math.fsum((a - b)**2 for p, t in zip(predicted, targets) for a, b in zip(p, t))


def reconstruction_metrics(predicted, targets, composite, *, mask_kind):
    """Per-pair normalized metrics with explicit label-conditioned/blind roles."""
    reconstruction_loss(predicted, targets)  # complete shape/range checks
    if mask_kind not in ("ground-truth", "predicted-top2"):
        raise ValueError("Explicit decoder mask provenance required")
    if len(composite) != 1296 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in composite):
        raise ValueError("Invalid normalized composite")
    ordered = math.fsum((p - t)**2 for row_p, row_t in zip(predicted, targets) for p, t in zip(row_p, row_t))
    swapped = math.fsum((p - t)**2 for row_p, row_t in zip(predicted, reversed(targets)) for p, t in zip(row_p, row_t))
    conservation = math.fsum((min(a + b, 1) - c)**2 for a, b, c in zip(*predicted, composite)) / 1296
    return {"maskKind": mask_kind, "labelConditionedSSE": ordered if mask_kind == "ground-truth" else None,
            "labelConditionedMSE": ordered / 2592 if mask_kind == "ground-truth" else None,
            "predictedMaskPermutationMSE": min(ordered, swapped) / 2592 if mask_kind == "predicted-top2" else None,
            "clippedMixtureConservationMSE": conservation}


def learning_rate(update):
    integer(update, "completed updates", 0, 2**53)
    return max(1e-6, .001 * .96**(update / 20000))


class Metrics:
    def __init__(self):
        self.pairs = self.both = self.one = self.missed = self.ties = 0
        self.true = [0] * 10
        self.predicted = [0] * 10
        self.tp = [0] * 10

    def add(self, scores, labels):
        margin(scores, labels, "paper-equations")  # shared shape/value checks
        order = sorted(range(10), key=lambda k: (-scores[k], k))
        predicted, truth = set(order[:2]), set(labels)
        hits = len(predicted & truth)
        self.pairs += 1
        self.both += hits == 2
        self.one += hits >= 1
        self.missed += 2 - hits
        self.ties += scores[order[1]] == scores[order[2]]
        for label in truth:
            self.true[label] += 1
        for label in predicted:
            self.predicted[label] += 1
        for label in truth & predicted:
            self.tp[label] += 1

    def result(self):
        if not self.pairs:
            raise ValueError("No observations; a zero denominator is not a score")
        return {"pairs": self.pairs, "bothCorrect": self.both, "atLeastOneCorrect": self.one,
                "missedDigits": self.missed, "pairError": (self.pairs - self.both) / self.pairs,
                "perDigitError": self.missed / (2 * self.pairs), "bothAccuracy": self.both / self.pairs,
                "atLeastOneAccuracy": self.one / self.pairs, "top2BoundaryTies": self.ties,
                "truePerClass": self.true[:], "predictedPerClass": self.predicted[:], "truePositivePerClass": self.tp[:]}


if __name__ == "__main__":
    print(json.dumps(Protocol().describe(), indent=2, allow_nan=False))
