"""Called only by the explicitly granted CPU reference gate; no native import here."""
from pathlib import Path
import copy


def run(torch, output):
    from paper_capsnet import tensor_squash
    from paper_engine import TorchEngine
    from paper_train import Trainer, evaluate_final
    from paper_protocol import Protocol, PairStream, Metrics, reconstruction_metrics
    from paper_data import batches
    from paper_state import Checkpoints
    from paper_protocol import digest

    class TinyFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(1296, 160)
            self.decoder = torch.nn.Linear(160, 1296)

        def reconstruct(self, capsules, labels):
            mask = torch.nn.functional.one_hot(labels, 10).to(capsules.dtype)
            return torch.sigmoid(self.decoder((capsules*mask.unsqueeze(-1)).flatten(1)))

        def forward(self, images, labels):
            capsules = tensor_squash(self.encoder(images.flatten(1)).reshape(-1, 10, 16))
            return torch.linalg.vector_norm(capsules, dim=-1), capsules, torch.stack(
                [self.reconstruct(capsules, labels[:, k]) for k in range(2)], dim=1)

    p = Protocol(pairs_per_anchor=1, batch_size=2, validation_anchors=2, validation_every_updates=2)
    labels = bytes([0, 1, 2, 3, 0, 1, 2, 3])
    images = bytes(i % 256 for i in range(len(labels)*784))
    binding = {k+"SHA256": digest(["cpu-native-fixture-not-MNIST", k])
               for k in ("protocol", "source", "data", "runtime")}
    def create(name):
        torch.manual_seed(20261107)
        engine = TorchEngine("cpu", fixture_model=TinyFixture())
        store = Checkpoints(output/name, binding)
        trainer = Trainer(p, PairStream(labels, "train", p, fixture=True), images, engine, store,
                          fixture=True, checkpoint_every=1)
        return trainer, engine, store
    def equal(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (tuple, list)):
            return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return type(a) is type(b) and a == b

    full, a, _ = create("native-full")
    full.run()
    expected = copy.deepcopy(a.snapshot())
    partial, b, _ = create("native-resume")
    partial.run(lambda: b.completed_updates == 1)
    resumed, c, store = create("native-resume")
    resumed.run()
    if not equal(expected, c.snapshot()) or full.state != resumed.state:
        raise RuntimeError("Actual tensor trainer/model/optimizer/RNG/metric resume differs")
    before = copy.deepcopy(c.snapshot())
    for mutation in ("moments", "precision", "dtype"):
        bad = copy.deepcopy(before)
        if mutation == "moments":
            bad["optimizer"]["state"].pop(0)
        elif mutation == "precision":
            bad["precision"] = "amp"
        else:
            key = next(iter(bad["model"]))
            bad["model"][key] = bad["model"][key].double()
        try:
            c.restore(bad)
        except ValueError:
            pass
        else:
            raise RuntimeError("Native altered checkpoint accepted: " + mutation)
        if not equal(before, c.snapshot()):
            raise RuntimeError("Refused checkpoint mutated current native state")

    stream = PairStream(bytes([0, 1, 2, 3]), "test", p, fixture=True)
    test_images = images[:4*784]
    first = next(batches(stream, test_images, p.batch_size))
    native = c.evaluate(iter([first]), "test")
    x, targets, y = c.tensors(first, "test")
    scalar = Metrics()
    gt_sse = blind_sse = gt_conservation = blind_conservation = 0.
    with torch.no_grad():
        scores, capsules, gt = c.model(x, y)
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :2]
        blind = torch.stack([c.model.reconstruct(capsules, order[:, k]) for k in range(2)], 1)
        for i, pair in enumerate(first["pairs"]):
            scalar.add(scores[i].tolist(), pair.labels)
            a = reconstruction_metrics(gt[i].tolist(), targets[i].tolist(), x[i].flatten().tolist(), mask_kind="ground-truth")
            b = reconstruction_metrics(blind[i].tolist(), targets[i].tolist(), x[i].flatten().tolist(), mask_kind="predicted-top2")
            gt_sse += a["labelConditionedSSE"]
            blind_sse += b["predictedMaskPermutationMSE"]*2592
            gt_conservation += a["clippedMixtureConservationMSE"]*1296
            blind_conservation += b["clippedMixtureConservationMSE"]*1296
    reference = scalar.result()
    for key in ("pairs", "bothCorrect", "atLeastOneCorrect", "missedDigits", "top2BoundaryTies",
                "truePerClass", "predictedPerClass", "truePositivePerClass"):
        if native[key] != reference[key]:
            raise RuntimeError("Native/scalar metric counts differ: " + key)
    for key, expected_sum in (("labelConditionedSSE", gt_sse), ("predictedMaskPermutationSSE", blind_sse),
                              ("groundTruthConservationSSE", gt_conservation), ("predictedConservationSSE", blind_conservation)):
        if abs(native[key]-expected_sum) > 1e-10:
            raise RuntimeError("Native/scalar reconstruction numerator differs: " + key)
    result = evaluate_final(p, stream, test_images, c, store, Checkpoints(output/"native-test", binding),
                            fixture=True, train_count=8, checkpoint_batches=1)
    if result["testPairsEvaluated"] != 4 or not equal(before, c.snapshot()):
        raise RuntimeError("Final-only fixture evaluation omitted data or changed model/optimizer/RNG")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU gate initialized CUDA")
    return {"passed": True, "syntheticOnly": True, "MNISTResults": False,
            "actualOptimizerUpdates": 4, "fixtureTestPairs": 4,
            "bitwiseTrainerModelOptimizerRNGResume": True,
            "invalidMomentsPrecisionDtypeRefusedBeforeMutation": True,
            "scalarTensorCountAndReconstructionNumerators": True,
            "evaluationModelOptimizerRNGUnchanged": True}
