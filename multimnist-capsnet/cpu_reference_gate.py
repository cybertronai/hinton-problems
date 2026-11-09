"""Future explicit CPU-native gate. NOT run by the stdlib fixture suite.

Run only in a separately approved, pinned, private Torch process. No MNIST
data, GPU discovery, downloads or training-result claims are made here.
"""
import argparse
import copy
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("Explicit CPU-only child environment required before importing Torch")
    if not (sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode):
        raise RuntimeError("Explicit isolated/no-site/no-bytecode interpreter required")
    if args.output.exists():
        raise FileExistsError("Fresh native proof output required")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from paper_capsnet import build_model, build_optimizer, paired_loss, tensor_route
    from paper_protocol import digest, learning_rate, margin, reconstruction_loss, route
    from paper_state import Checkpoints
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_default_device("cpu")
    torch.manual_seed(20261107)
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    checks = {}

    votes = torch.tensor([[[[.2, -.1], [.1, .4]], [[-.4, .3], [.2, -.2]], [[.6, .1], [-.1, .5]]]],
                         dtype=torch.float64, requires_grad=True)
    bias = torch.tensor([[.1, .1], [.1, .1]], dtype=torch.float64, requires_grad=True)
    expected, _ = route(votes.detach().tolist()[0], bias=bias.detach().tolist())
    actual = tensor_route(votes, bias)
    torch.testing.assert_close(actual[0], torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-14)
    if not torch.autograd.gradcheck(lambda v, b: tensor_route(v, b), (votes, bias), eps=1e-6, atol=1e-5, rtol=1e-4):
        raise RuntimeError("Fully differentiated routing finite-difference mismatch")
    checks["scalarRoutingAndFiniteDifference"] = True

    scores = torch.tensor([[.2, .5, .7, .1, .2, .3, .4, .5, .1, .2]], dtype=torch.float64)
    labels = torch.tensor([[2, 7]], dtype=torch.int64)
    predicted = torch.zeros(1, 2, 1296, dtype=torch.float64)
    targets = torch.ones_like(predicted)
    for profile in ("author2017", "paper-equations"):
        loss, parts = paired_loss(scores, labels, predicted, targets, profile)
        classification = margin(scores[0].tolist(), [2, 7], profile)
        reconstruction = reconstruction_loss(predicted[0].tolist(), targets[0].tolist())
        torch.testing.assert_close(parts["classification"], torch.tensor(classification, dtype=torch.float64), rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(loss, torch.tensor(classification + reconstruction, dtype=torch.float64), rtol=1e-12, atol=1e-14)
    checks["lossReductionsBothConditions"] = True

    parameters = [torch.nn.Parameter(torch.tensor([.2, -.7], dtype=torch.float64))]
    optimizer = build_optimizer(parameters)
    m, v, independent = [0., 0.], [0., 0.], [.2, -.7]
    for update in range(5):
        gradient = [.3 / (update + 1), -.2 * (update + 1)]
        parameters[0].grad = torch.tensor(gradient, dtype=torch.float64)
        optimizer.step()
        for k in range(2):
            m[k] = .9 * m[k] + .1 * gradient[k]
            v[k] = .999 * v[k] + .001 * gradient[k]**2
            # Written as bias-corrected moments with the equivalent TF epsilon.
            mt, vt = m[k] / (1 - .9**(update + 1)), v[k] / (1 - .999**(update + 1))
            independent[k] -= learning_rate(update) * mt / (math.sqrt(vt) + 1e-8 / math.sqrt(1 - .999**(update + 1)))
        torch.testing.assert_close(parameters[0], torch.tensor(independent, dtype=torch.float64), rtol=1e-13, atol=1e-14)
    checks["explicitAdamFormula"] = True

    saved = io.BytesIO()
    torch.save({"parameter": parameters[0].detach().clone(), "optimizer": copy.deepcopy(optimizer.state_dict()),
                "rng": torch.get_rng_state(), "nextCursor": 640}, saved)
    # Fixture identity is explicit; this is not an MNIST admission/checkpoint.
    binding = {key + "SHA256": digest(["native-cpu-fixture", key]) for key in ("protocol", "source", "data", "runtime")}
    store = Checkpoints(args.output / "fixture-checkpoints", binding)
    store.commit(saved.getvalue(), 640, 5)
    state = torch.load(io.BytesIO(store.read()[0]), map_location="cpu", weights_only=True)
    resumed = torch.nn.Parameter(state["parameter"].clone())
    resumed_optimizer = build_optimizer([resumed])
    resumed_optimizer.load_state_dict(state["optimizer"])
    for parameter, opt in ((parameters[0], optimizer), (resumed, resumed_optimizer)):
        parameter.grad = torch.tensor([.05, -.03], dtype=torch.float64)
        opt.step()
    torch.testing.assert_close(resumed, parameters[0], rtol=0., atol=0.)
    if optimizer.completed_updates != resumed_optimizer.completed_updates or state["nextCursor"] != 640:
        raise RuntimeError("Optimizer/cursor continuation mismatch")
    torch.set_rng_state(state["rng"])
    draw = torch.rand(10)
    torch.set_rng_state(state["rng"])
    torch.testing.assert_close(torch.rand(10), draw, rtol=0., atol=0.)
    checks["serializedOptimizerCursorAndRNGResume"] = True

    # Full-width shapes and actual parameter/gradient paths, one synthetic row.
    # Keep profiles sequential to bound host memory; this is not a throughput pilot.
    for profile in ("author2017", "paper-equations"):
        model = build_model(profile).cpu()
        image = torch.linspace(0, 1, 1296).reshape(1, 1, 36, 36)
        paired = torch.tensor([[1, 6]], dtype=torch.int64)
        scores, capsules, recon = model(image, paired)
        if tuple(capsules.shape) != (1, 10, 16) or tuple(recon.shape) != (1, 2, 1296):
            raise RuntimeError("Full-width forward shape mismatch")
        loss, loss_parts = paired_loss(scores, paired, recon, image.flatten(1)[:, None, :].expand(-1, 2, -1), profile)
        loss.backward()
        missing = [name for name, parameter in model.named_parameters()
                   if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())]
        if missing or not bool(torch.isfinite(loss)):
            raise RuntimeError("Full-width finite-gradient gate failed: " + repr(missing))
        updates = 0
        if profile == "author2017":
            full_optimizer = build_optimizer(model.parameters())
            full_optimizer.step()
            if full_optimizer.completed_updates != 1 or any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise RuntimeError("Full-width author optimizer step failed")
            updates = 1
            del full_optimizer
        checks[profile + "FullWidth"] = {"parameters": sum(p.numel() for p in model.parameters()),
                                           "capsules": list(capsules.shape), "reconstruction": list(recon.shape),
                                           "finiteGradientParameters": len(list(model.parameters())),
                                           "actualOptimizerUpdates": updates}
        del model, image, paired, scores, capsules, recon, loss, loss_parts
        gc.collect()

    from cpu_trainer_gate import run as trainer_checks
    checks["actualFiniteTrainer"] = trainer_checks(torch, args.output)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU reference gate initialized CUDA")

    source = Path(__file__).resolve().parent
    result = {"passed": True, "syntheticOnly": True, "MNISTResults": False,
              "python": sys.version, "torch": torch.__version__, "device": "cpu",
              "cudaInitialized": False, "isolated": sys.flags.isolated,
              "noSite": sys.flags.no_site, "dontWriteBytecode": sys.dont_write_bytecode,
              "seconds": time.monotonic() - started, "checks": checks,
              "source": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob("paper_*.py")}}
    with (args.output / "RESULT.json").open("xb") as handle:
        handle.write((json.dumps(result, indent=2, allow_nan=False) + "\n").encode())
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
