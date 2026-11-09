"""Finite, resumable control flow. Import is stdlib-only, never a device choice.

The production CLI is deliberately separate from execution authority: its caller
must authenticate source/data/runtime and hold the applicable resource grant.
"""
from contextlib import contextmanager
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_protocol import Protocol, PairStream, Permutation, canonical, digest, integer, render
from paper_data import batches
from paper_state import Checkpoints


class StopRequested(Exception):
    """Only raised at an explicit safe boundary; never disguises model errors."""


COUNT_KEYS = ("bothCorrect", "atLeastOneCorrect", "missedDigits", "top2BoundaryTies")
CLASS_KEYS = ("truePerClass", "predictedPerClass", "truePositivePerClass")
SSE_KEYS = ("labelConditionedSSE", "predictedMaskPermutationSSE",
            "groundTruthConservationSSE", "predictedConservationSSE")


class Aggregate:
    def __init__(self, state=None):
        self.state = dict(pairs=0, **{k: 0 for k in COUNT_KEYS},
                          **{k: [0] * 10 for k in CLASS_KEYS}, **{k: 0. for k in SSE_KEYS})
        if state is not None:
            self.add(state, allow_empty=True)

    def add(self, row, *, allow_empty=False):
        if not isinstance(row, dict) or set(row) != set(self.state):
            raise ValueError("Complete metric numerators required")
        n = integer(row["pairs"], "metric pairs", 0 if allow_empty else 1, 100000000)
        for key in COUNT_KEYS:
            integer(row[key], key, 0, 2 * n)
        both, one = row["bothCorrect"], row["atLeastOneCorrect"]
        if not both <= one <= n or row["top2BoundaryTies"] > n or row["missedDigits"] != 2*n-one-both:
            raise ValueError("Inconsistent pair/digit numerators")
        for key in CLASS_KEYS:
            if type(row[key]) is not list or len(row[key]) != 10:
                raise ValueError("Ten class counts required")
            for value in row[key]:
                integer(value, key, 0, n)
        if sum(row["truePerClass"]) != 2*n or sum(row["predictedPerClass"]) != 2*n or sum(row["truePositivePerClass"]) != both+one:
            raise ValueError("Inconsistent per-class denominators")
        if any(tp > min(t, p) for tp, t, p in zip(row["truePositivePerClass"], row["truePerClass"], row["predictedPerClass"])):
            raise ValueError("Impossible class true positives")
        for key in SSE_KEYS:
            value = row[key]
            cap = n * (2592 if key in SSE_KEYS[:2] else 1296)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= cap:
                raise ValueError("Invalid complete reconstruction numerator")
        for key in ("pairs", *COUNT_KEYS, *SSE_KEYS):
            self.state[key] += row[key]
        for key in CLASS_KEYS:
            self.state[key] = [a+b for a, b in zip(self.state[key], row[key])]

    def report(self):
        s = json.loads(canonical(self.state))
        n = s["pairs"]
        if not n:
            raise ValueError("Zero denominator is not an evaluation")
        return {"numerators": s, "denominators": {"pairs": n, "digits": 2*n,
                "pairedPixels": 2592*n, "canvasPixels": 1296*n},
                "pairError": (n-s["bothCorrect"])/n, "perDigitError": s["missedDigits"]/(2*n),
                "bothAccuracy": s["bothCorrect"]/n, "atLeastOneAccuracy": s["atLeastOneCorrect"]/n,
                "labelConditionedMSE": s["labelConditionedSSE"]/(2592*n),
                "predictedMaskPermutationMSE": s["predictedMaskPermutationSSE"]/(2592*n),
                "groundTruthConservationMSE": s["groundTruthConservationSSE"]/(1296*n),
                "predictedConservationMSE": s["predictedConservationSSE"]/(1296*n)}


def diagnostic_batches(stream, images):
    anchors = Permutation(len(stream.labels), [stream.protocol.seed, "train-diagnostic-anchors"])
    n = stream.protocol.validation_anchors
    if n > len(stream.labels):
        raise ValueError("Diagnostic anchors exceed the admitted split")
    for start in range(0, n, stream.protocol.batch_size):
        pairs = [stream.validation(anchors.at(i)) for i in range(start, min(start+stream.protocol.batch_size, n))]
        rows = [render(images[p.anchor*784:(p.anchor+1)*784], images[p.partner*784:(p.partner+1)*784], p) for p in pairs]
        yield {"pairs": pairs, "images": b"".join(r[0] for r in rows),
               "targets": tuple(b"".join(r[1][k] for r in rows) for k in range(2))}


@contextmanager
def session(store, phase):
    path = store.root / "SESSION.lock"
    with path.open("xb") as out:
        out.write(canonical({"phase": phase, "binding": store.binding, "pid": os.getpid()}))
        out.flush()
        os.fsync(out.fileno())
    normal = False
    try:
        yield
        normal = True
    finally:
        if normal:
            path.unlink()
        # An unexpected exception/kill is evidence, never an automatically stolen lock.


def validate_state(state, protocol, total):
    if set(state) != {"completedPairs", "completedUpdates", "completedExposures", "online", "diagnostics", "bestUpdate", "bestRank"}:
        raise ValueError("Incomplete trainer cursor/selection state")
    n = integer(state["completedPairs"], "completed pairs", 0, total)
    u = integer(state["completedUpdates"], "completed updates", 0, math.ceil(total/protocol.batch_size))
    if u != math.ceil(n/protocol.batch_size) or (n != total and n % protocol.batch_size):
        raise ValueError("Cursor/update/batch relationship differs")
    if type(state["completedExposures"]) is not int or state["completedExposures"] != int(n == total):
        raise ValueError("Single-exposure cursor differs")
    online = state["online"]
    if set(online) != {"pairs", "classificationSum", "reconstructionSum"} or online["pairs"] != n:
        raise ValueError("Online observation denominator differs")
    if any(type(online[k]) not in (float, int) or not math.isfinite(online[k]) or online[k] < 0
           for k in ("classificationSum", "reconstructionSum")):
        raise ValueError("Nonfinite online loss")
    schedule = list(range(protocol.validation_every_updates, u+1, protocol.validation_every_updates))
    if not isinstance(state["diagnostics"], list) or [r["update"] for r in state["diagnostics"]] != schedule:
        raise ValueError("Missing, duplicate or out-of-order train-only diagnostics")
    ranks = []
    for row in state["diagnostics"]:
        if set(row) != {"update", "metrics"}:
            raise ValueError("Unexpected diagnostic state")
        metrics = Aggregate(row["metrics"]).state
        if metrics["pairs"] != protocol.validation_anchors:
            raise ValueError("Incomplete diagnostic anchors")
        ranks.append(([metrics["pairs"]-metrics["bothCorrect"], metrics["labelConditionedSSE"]], row["update"]))
    best = min(ranks, key=lambda x: (x[0], x[1])) if ranks else (None, None)
    if (state["bestRank"], state["bestUpdate"]) != best:
        raise ValueError("Diagnostic best selection differs")


class Trainer:
    def __init__(self, protocol, stream, images, backend, store, *, fixture=False, checkpoint_every=1000):
        if type(fixture) is not bool:
            raise ValueError("Explicit fixture boolean required")
        if protocol.exposures != 1:
            raise ValueError("This finite trainer implements exactly one exposure")
        if protocol != Protocol() and not fixture:
            raise ValueError("Production requires the approved explicit default author2017 protocol")
        if not fixture and (store.binding["protocolSHA256"] != protocol.sha256 or getattr(backend, "kind", None) != "full-width-author2017"):
            raise ValueError("Production protocol/model binding differs")
        if not isinstance(stream, PairStream) or stream.split != "train" or stream.protocol != protocol:
            raise ValueError("Matching training-only stream required")
        if len(images) != len(stream.labels)*784 or (not fixture and stream.count != 60000000):
            raise ValueError("Complete training source required")
        self.p, self.stream, self.images, self.engine, self.store = protocol, stream, images, backend, store
        self.fixture = fixture
        self.cadence = integer(checkpoint_every, "checkpoint cadence", 1, 10000)
        self.total, self.updates = stream.count, math.ceil(stream.count/protocol.batch_size)
        self.state = {"completedPairs": 0, "completedUpdates": 0, "completedExposures": 0,
                      "online": {"pairs": 0, "classificationSum": 0., "reconstructionSum": 0.},
                      "diagnostics": [], "bestUpdate": None, "bestRank": None}
        self.latest = None

    def validate(self, state):
        validate_state(state, self.p, self.total)

    def restore(self):
        current = self.store.read()
        if current is None:
            if self.engine.completed_updates != 0:
                raise ValueError("New run must start with a new optimizer")
            return
        raw, entry = current
        payload = self.engine.decode(raw)  # only authenticated owned bytes; native uses weights_only=True
        if set(payload) != {"kind", "binding", "protocol", "fixture", "state", "backend"} or payload["kind"] != "capsnet-training-v1":
            raise ValueError("Wrong checkpoint phase/schema")
        if payload["binding"] != self.store.binding or payload["protocol"] != asdict(self.p) or payload["fixture"] is not self.fixture:
            raise ValueError("Checkpoint scientific identity differs")
        self.validate(payload["state"])
        if (entry["completedPairs"], entry["completedUpdates"]) != (payload["state"]["completedPairs"], payload["state"]["completedUpdates"]):
            raise ValueError("Outer and serialized cursors differ")
        best = self.store.read("best")
        if (best is None) != (payload["state"]["bestUpdate"] is None) or (best and best[1]["completedUpdates"] != payload["state"]["bestUpdate"]):
            raise ValueError("Retained diagnostic-best pointer differs")
        self.engine.restore(payload["backend"])
        if self.engine.completed_updates != entry["completedUpdates"]:
            raise ValueError("Optimizer update count differs")
        self.state, self.latest = payload["state"], entry

    def save(self, *, best=False):
        self.validate(self.state)
        if self.engine.completed_updates != self.state["completedUpdates"]:
            raise ValueError("Model/optimizer cursor mismatch; do not commit")
        if self.latest and self.latest["completedUpdates"] == self.state["completedUpdates"]:
            return self.latest
        raw = self.engine.encode({"kind": "capsnet-training-v1", "binding": self.store.binding,
             "protocol": asdict(self.p), "fixture": self.fixture, "state": self.state,
             "backend": self.engine.snapshot()})
        self.latest = self.store.commit(raw, self.state["completedPairs"], self.state["completedUpdates"], best=best)
        return self.latest

    def run(self, stop=lambda: False):
        with session(self.store, "train"):
            self.restore()
            self.save()  # authenticated initial state also makes a zero-update stop resumable
            try:
                for batch in batches(self.stream, self.images, self.p.batch_size, start=self.state["completedPairs"]):
                    if stop():
                        raise StopRequested()
                    observation = self.engine.step(batch)
                    n = len(batch["pairs"])
                    if set(observation) != {"pairs", "classificationSum", "reconstructionSum"} or observation["pairs"] != n:
                        raise ValueError("Incomplete step observation")
                    for k in ("classificationSum", "reconstructionSum"):
                        if type(observation[k]) not in (float, int) or not math.isfinite(observation[k]) or observation[k] < 0:
                            raise ValueError("Invalid step loss")
                        self.state["online"][k] += observation[k]
                    self.state["online"]["pairs"] += n
                    self.state["completedPairs"] = batch["nextCursor"]
                    self.state["completedExposures"] = int(batch["nextCursor"] == self.total)
                    self.state["completedUpdates"] += 1
                    u = self.state["completedUpdates"]
                    if self.engine.completed_updates != u:
                        raise ValueError("Optimizer did not complete exactly one update")
                    best = False
                    if u % self.p.validation_every_updates == 0:
                        result = Aggregate(self.engine.evaluate(diagnostic_batches(self.stream, self.images), "train-diagnostic", stop)).state
                        if result["pairs"] != self.p.validation_anchors or self.engine.completed_updates != u:
                            raise ValueError("Diagnostic incomplete or optimizer changed during evaluation")
                        rank = [result["pairs"]-result["bothCorrect"], result["labelConditionedSSE"]]
                        best = self.state["bestRank"] is None or rank < self.state["bestRank"]
                        self.state["diagnostics"].append({"update": u, "metrics": result})
                        if best:
                            self.state.update(bestRank=rank, bestUpdate=u)
                    if best or u % self.cadence == 0:
                        self.save(best=best)
                self.save()
                return {"status": "trained", "completedPairs": self.total, "completedUpdates": self.updates,
                        "finalCheckpoint": self.latest, "testPairsEvaluated": 0, "syntheticOnly": self.fixture,
                        "diagnosticEvaluations": len(self.state["diagnostics"]),
                        "uniqueDiagnosticPairs": self.p.validation_anchors if self.state["diagnostics"] else 0,
                        "diagnosticPairObservations": sum(r["metrics"]["pairs"] for r in self.state["diagnostics"]),
                        "onlineTraining": {"pairs": self.total,
                            "meanClassificationLoss": self.state["online"]["classificationSum"]/self.total,
                            "meanReconstructionLoss": self.state["online"]["reconstructionSum"]/self.total,
                            "isFinalModelEvaluation": False}}
            except StopRequested:
                # Never checkpoint a half-completed diagnostic; resume from the
                # previous authenticated state and disclose discarded work.
                scheduled = self.state["completedUpdates"] // self.p.validation_every_updates
                if len(self.state["diagnostics"]) == scheduled:
                    self.save()
                return {"status": "stopped", "committedPairs": self.latest["completedPairs"],
                        "committedUpdates": self.latest["completedUpdates"],
                        "uncommittedPairs": self.state["completedPairs"]-self.latest["completedPairs"],
                        "testPairsEvaluated": 0, "syntheticOnly": self.fixture}


def final_checkpoint(store, engine, protocol, *, fixture=False, train_count=60000000):
    if type(fixture) is not bool:
        raise ValueError("Explicit fixture boolean required")
    if not fixture and (store.binding["protocolSHA256"] != protocol.sha256 or getattr(engine, "kind", None) != "full-width-author2017"):
        raise ValueError("Production protocol/model binding differs")
    if (not fixture and (protocol != Protocol() or train_count != 60000000)) or (store.root / "SESSION.lock").exists():
        raise ValueError("Approved complete protocol and inactive trainer required")
    raw, entry = store.read() or (None, None)
    if raw is None:
        raise ValueError("No final training checkpoint")
    payload = engine.decode(raw)
    expected = math.ceil(train_count/protocol.batch_size)
    if payload.get("kind") != "capsnet-training-v1" or payload.get("binding") != store.binding or payload.get("protocol") != asdict(protocol) or payload.get("fixture") is not fixture:
        raise ValueError("Final checkpoint identity differs")
    state = payload["state"]
    validate_state(state, protocol, train_count)
    if (entry["completedPairs"], entry["completedUpdates"], state["completedPairs"], state["completedUpdates"]) != (train_count, expected, train_count, expected):
        raise ValueError("Official test is final-checkpoint-only")
    engine.restore(payload["backend"])
    if engine.completed_updates != expected:
        raise ValueError("Final optimizer state differs")
    return entry


def evaluate_final(protocol, stream, images, engine, training_store, evaluation_store,
                   *, fixture=False, train_count=60000000, checkpoint_batches=100, stop=lambda: False):
    if type(fixture) is not bool:
        raise ValueError("Explicit fixture boolean required")
    if (protocol != Protocol() and not fixture) or stream.split != "test" or stream.protocol != protocol:
        raise ValueError("Explicit approved test-only stream required")
    if not fixture and stream.count != 10000000:
        raise ValueError("Complete official test denominator required")
    # Caller must verify final_checkpoint BEFORE opening test source bytes.
    entry = final_checkpoint(training_store, engine, protocol, fixture=fixture, train_count=train_count)
    if evaluation_store.binding != training_store.binding:
        raise ValueError("Evaluation binding differs")
    integer(checkpoint_batches, "evaluation checkpoint cadence", 1, 1000)
    with session(evaluation_store, "official-test"):
        aggregate, cursor, batch_count = Aggregate(), 0, 0
        saved = evaluation_store.read()
        if saved:
            state = json.loads(saved[0])
            if set(state) != {"kind", "binding", "finalCheckpointSHA256", "metrics"} or state["kind"] != "capsnet-official-test-v1" or state["binding"] != training_store.binding or state["finalCheckpointSHA256"] != entry["sha256"]:
                raise ValueError("Evaluation cursor/model identity differs")
            aggregate = Aggregate(state["metrics"])
            cursor, batch_count = saved[1]["completedPairs"], saved[1]["completedUpdates"]
            if cursor != aggregate.state["pairs"] or cursor > stream.count or batch_count != math.ceil(cursor/protocol.batch_size) or (cursor != stream.count and cursor % protocol.batch_size):
                raise ValueError("Evaluation cursor/batch relationship differs")
        def save():
            evaluation_store.commit(canonical({"kind": "capsnet-official-test-v1", "binding": training_store.binding,
                "finalCheckpointSHA256": entry["sha256"], "metrics": aggregate.state}), cursor, batch_count)
        last_saved = cursor
        try:
            for batch in batches(stream, images, protocol.batch_size, start=cursor):
                if stop():
                    raise StopRequested()
                value = engine.evaluate(iter([batch]), "test", stop)
                if value["pairs"] != len(batch["pairs"]) or engine.completed_updates != entry["completedUpdates"]:
                    raise ValueError("Test omitted rows or mutated optimizer")
                aggregate.add(value)
                cursor, batch_count = batch["nextCursor"], batch_count+1
                if batch_count % checkpoint_batches == 0:
                    save()
                    last_saved = cursor
            if cursor != last_saved:
                save()
            return {"status": "evaluated", "modelCheckpointSHA256": entry["sha256"],
                    "selection": "final-checkpoint-only", "syntheticOnly": fixture,
                    "trainingOptimizerUpdates": entry["completedUpdates"],
                    "testPairsEvaluated": cursor, "testBatchesEvaluated": batch_count,
                    "metrics": aggregate.report()}
        except StopRequested:
            if cursor != last_saved:
                save()
            return {"status": "stopped", "testPairsEvaluated": cursor, "testBatchesEvaluated": batch_count,
                    "trainingOptimizerUpdates": entry["completedUpdates"], "syntheticOnly": fixture}


def main():
    import argparse
    import hashlib
    from paper_data import read_split
    from paper_engine import TorchEngine
    from paper_state import ordinary
    parser = argparse.ArgumentParser(description="Fixed author2017 train, then separate final-only official test")
    parser.add_argument("--phase", choices=("train", "test"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.absolute()
    for item in reversed([output, *output.parents]):
        ordinary(item, directory=True)  # caller creates its explicitly owned output, no recursive mkdir
    source = Path(__file__).resolve().parent
    binding = {"protocolSHA256": Protocol().sha256,
               "sourceSHA256": digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(source.glob("paper_*.py"))}),
               "dataSHA256": args.manifest_sha256, "runtimeSHA256": args.runtime_sha256}
    training = Checkpoints(output / "training", binding)
    engine = TorchEngine(args.device)
    def stop():
        flag = output / "STOP"
        if flag.exists() or flag.is_symlink():
            ordinary(flag)
            if flag.stat().st_size != 0:
                raise ValueError("STOP must be an ordinary empty owned file")
            return True
        return False
    if args.phase == "test":
        final_checkpoint(training, engine, Protocol())  # BEFORE opening any official test source bytes
    images, labels = read_split(args.data_root, args.manifest_sha256, "train" if args.phase == "train" else "test")
    stream = PairStream(labels, "train" if args.phase == "train" else "test", Protocol())
    if args.phase == "train":
        result = Trainer(Protocol(), stream, images, engine, training).run(stop)
    else:
        result = evaluate_final(Protocol(), stream, images, engine, training,
                                Checkpoints(output / "official-test", binding), stop=stop)
    result.update(binding=binding, protocol=Protocol().describe(), actualDevice=args.device,
                  onlineTrainingMetricsAreNotFinalModelScores=True)
    if result["status"] == "evaluated":
        raw = canonical(result) + b"\n"
        path = output / "TEST_RESULT.json"
        if path.exists():
            ordinary(path)
            if path.read_bytes() != raw:
                raise ValueError("Existing final result differs; preserve and reconcile")
        else:
            with path.open("xb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
