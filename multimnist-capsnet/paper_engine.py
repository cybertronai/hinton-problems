"""Explicit native backend. Importing this module does NOT import Torch."""
import io
import os
import random


class TorchEngine:
    def __init__(self, device, *, fixture_model=None):
        if device not in ("cpu", "cuda:0"):
            raise ValueError("Choose cpu or the externally granted single cuda:0 explicitly")
        if device == "cpu" and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            raise ValueError("CPU environment must hide CUDA before import")
        if device == "cuda:0" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            raise ValueError("Deterministic CUDA workspace must be bound before import")
        import torch
        from paper_capsnet import build_model, build_optimizer
        self.torch, self.device = torch, device
        torch.set_default_device("cpu")
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(20261107)
        random.seed(20261107)
        self.kind = "synthetic-native-fixture" if fixture_model is not None else "full-width-author2017"
        self.model = (fixture_model if fixture_model is not None else build_model("author2017")).to(device=device, dtype=torch.float32)
        self.optimizer = build_optimizer(self.model.parameters())

    @property
    def completed_updates(self):
        return self.optimizer.completed_updates

    def tensors(self, batch, purpose):
        torch = self.torch
        n = len(batch["pairs"])
        expected = "train" if purpose == "train" else purpose
        if not 1 <= n <= 128 or any(p.split != expected for p in batch["pairs"]):
            raise ValueError("Wrong split/purpose or incomplete batch")
        if len(batch["images"]) != n*1296 or any(len(row) != n*1296 for row in batch["targets"]) or len(batch["targets"]) != 2:
            raise ValueError("Complete composite and paired targets required")
        def pixels(raw):
            return torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device=self.device, dtype=torch.float32).div_(255)
        images = pixels(batch["images"]).reshape(n, 1, 36, 36)
        targets = torch.stack([pixels(row).reshape(n, 1296) for row in batch["targets"]], dim=1)
        labels = torch.tensor([p.labels for p in batch["pairs"]], dtype=torch.int64, device=self.device)
        return images, targets, labels

    def step(self, batch):
        from paper_capsnet import paired_loss
        images, targets, labels = self.tensors(batch, "train")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        scores, _, predicted = self.model(images, labels)
        loss, parts = paired_loss(scores, labels, predicted, targets, "author2017")
        loss.backward()
        self.optimizer.step()
        n = len(batch["pairs"])
        return {"pairs": n, "classificationSum": float(parts["classification"].detach().cpu())*n,
                "reconstructionSum": float(parts["reconstruction"].detach().cpu())*n}

    def evaluate(self, rows, purpose, stop=lambda: False):
        from paper_train import Aggregate, StopRequested
        if purpose not in ("train-diagnostic", "test"):
            raise ValueError("Explicit diagnostic or official-test purpose required")
        torch, aggregate = self.torch, Aggregate()
        previous = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch in rows:
                    if stop():
                        raise StopRequested()
                    images, targets, labels = self.tensors(batch, purpose)
                    scores, capsules, gt = self.model(images, labels)
                    # Stable descending ordering preserves class-index tie order.
                    order = torch.argsort(scores, dim=1, descending=True, stable=True)
                    predicted_labels = order[:, :2]
                    blind = torch.stack([self.model.reconstruct(capsules, predicted_labels[:, k]) for k in range(2)], dim=1)
                    if any(not bool(torch.isfinite(x).all()) for x in (scores, gt, blind)):
                        raise ValueError("Nonfinite evaluation")
                    truth = torch.nn.functional.one_hot(labels, 10).sum(1)
                    prediction = torch.nn.functional.one_hot(predicted_labels, 10).sum(1)
                    tp = truth*prediction
                    hits = tp.sum(1)
                    # Metrics use FP64 numerators over FP32 model outputs, not
                    # a differently normalized loss or oracle decoder masking.
                    target64, gt64, blind64 = targets.double(), gt.double(), blind.double()
                    ordered = (blind64-target64).square().sum((1, 2))
                    swapped = (blind64-target64.flip(1)).square().sum((1, 2))
                    flat = images.flatten(1).double()
                    aggregate.add({"pairs": len(batch["pairs"]),
                        "bothCorrect": int((hits == 2).sum()), "atLeastOneCorrect": int((hits >= 1).sum()),
                        "missedDigits": int((2-hits).sum()),
                        "top2BoundaryTies": int((scores.gather(1, order[:, 1:2]) == scores.gather(1, order[:, 2:3])).sum()),
                        "truePerClass": truth.sum(0).cpu().tolist(),
                        "predictedPerClass": prediction.sum(0).cpu().tolist(),
                        "truePositivePerClass": tp.sum(0).cpu().tolist(),
                        "labelConditionedSSE": float((gt64-target64).square().sum()),
                        "predictedMaskPermutationSSE": float(torch.minimum(ordered, swapped).sum()),
                        "groundTruthConservationSSE": float((gt64.sum(1).clamp(0, 1)-flat).square().sum()),
                        "predictedConservationSSE": float((blind64.sum(1).clamp(0, 1)-flat).square().sum())})
        finally:
            self.model.train(previous)
        return aggregate.state

    def snapshot(self):
        torch = self.torch
        state = {"kind": self.kind, "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "cpuRNG": torch.get_rng_state(), "cudaRNG": torch.cuda.get_rng_state(self.device) if self.device != "cpu" else None,
                "pythonRNG": random.getstate(), "device": self.device, "trainingMode": self.model.training,
                "precision": "fp32-tf32-off-no-amp", "torchVersion": str(torch.__version__)}
        self.validate_snapshot(state)
        return state

    def validate_snapshot(self, state):
        torch = self.torch
        required = {"kind", "model", "optimizer", "cpuRNG", "cudaRNG", "pythonRNG", "device", "trainingMode", "precision", "torchVersion"}
        if set(state) != required or (state["kind"], state["device"], state["precision"], state["torchVersion"]) != (self.kind, self.device, "fp32-tf32-off-no-amp", str(torch.__version__)) or type(state["trainingMode"]) is not bool:
            raise ValueError("Native checkpoint/runtime/precision differs")
        current = self.model.state_dict()
        if set(state["model"]) != set(current):
            raise ValueError("Model keys differ")
        def tensor(value, expected):
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.shape != expected.shape or value.dtype != expected.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError("Wrong/nonfinite tensor state")
        for key, expected in current.items():
            tensor(state["model"][key], expected)
        opt = state["optimizer"]
        if set(opt) != {"state", "param_groups", "completed_updates"}:
            raise ValueError("Incomplete optimizer state")
        from paper_protocol import integer
        u = integer(opt["completed_updates"], "optimizer updates", 0, 468750)
        parameters = list(self.model.parameters())
        groups = opt["param_groups"]
        if len(groups) != 1 or set(groups[0]) != {"params", "beta1", "beta2", "eps"} or groups[0]["params"] != list(range(len(parameters))) or [groups[0][k] for k in ("beta1", "beta2", "eps")] != [.9, .999, 1e-8]:
            raise ValueError("Optimizer parameter/formula identity differs")
        if set(opt["state"]) != (set(range(len(parameters))) if u else set()):
            raise ValueError("Missing/excess optimizer moments")
        for i, parameter in enumerate(parameters):
            if u:
                if set(opt["state"][i]) != {"m", "v"}:
                    raise ValueError("Incomplete optimizer moments")
                for value in opt["state"][i].values():
                    tensor(value, parameter)
                if bool((opt["state"][i]["v"] < 0).any()):
                    raise ValueError("Negative second moment")
        cpu_rng = state["cpuRNG"]
        if not isinstance(cpu_rng, torch.Tensor) or cpu_rng.dtype != torch.uint8 or cpu_rng.shape != torch.get_rng_state().shape:
            raise ValueError("CPU RNG state differs")
        if self.device == "cpu" and state["cudaRNG"] is not None:
            raise ValueError("CUDA state in CPU checkpoint")
        random.Random().setstate(state["pythonRNG"])

    def restore(self, state):
        self.validate_snapshot(state)
        torch, opt = self.torch, state["optimizer"]
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(opt)
        torch.set_rng_state(state["cpuRNG"])
        if self.device == "cpu":
            if state["cudaRNG"] is not None:
                raise ValueError("CUDA state in CPU checkpoint")
        else:
            torch.cuda.set_rng_state(state["cudaRNG"], self.device)
        random.setstate(state["pythonRNG"])
        self.model.train(state["trainingMode"])

    def encode(self, payload):
        out = io.BytesIO()
        self.torch.save(payload, out)
        data = out.getvalue()
        if len(data) > 256*1024**2:
            raise ValueError("Checkpoint payload exceeds finite budget")
        return data

    def decode(self, payload):
        if type(payload) is not bytes or not 0 < len(payload) <= 256*1024**2:
            raise ValueError("Bounded authenticated payload required")
        return self.torch.load(io.BytesIO(payload), weights_only=True, map_location="cpu")
