"""Full-width independent PyTorch implementation (native gates still required).

Importing this file does not import Torch or create a device/model. Construction
is an explicit later operation. There is no downloader or automatic GPU choice.
"""
from paper_protocol import Protocol


def tensor_squash(vector):
    import torch
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector * (norm / (1 + norm.square()))


def tensor_route(votes, bias=None, iterations=3):
    import torch
    if votes.ndim != 4 or iterations != 3:
        raise ValueError("Expected B,input,output,atom votes and three routing iterations")
    logits = votes.new_zeros(votes.shape[:3])
    for step in range(iterations):
        coefficients = torch.softmax(logits, dim=2)
        pre = (coefficients.unsqueeze(-1) * votes).sum(dim=1)
        if bias is not None:
            pre = pre + bias
        capsules = tensor_squash(pre)
        if step != iterations - 1:
            # Fully differentiated routing, including the agreement updates.
            logits = logits + (votes * capsules.unsqueeze(1)).sum(dim=-1)
    return capsules


def build_model(profile="author2017"):
    import torch
    from torch import nn
    Protocol(profile=profile)  # reject an unmarked hybrid condition

    class FullCapsNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.profile = profile
            self.conv1 = nn.Conv2d(1, 256, 9)
            self.primary_conv = nn.Conv2d(256, 256, 9, stride=2, bias=False)
            self.primary_bias = nn.Parameter(torch.full((256,), .1))
            self.transforms = nn.Parameter(torch.empty(3200, 10, 16, 8))
            self.digit_bias = nn.Parameter(torch.full((10, 16), .1)) if profile == "author2017" else None
            self.decoder = nn.Sequential(nn.Linear(160, 512), nn.ReLU(), nn.Linear(512, 1024),
                                         nn.ReLU(), nn.Linear(1024, 1296), nn.Sigmoid())
            for layer in self.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear)):
                    std = .05 if layer is self.conv1 else .1
                    nn.init.trunc_normal_(layer.weight, mean=0., std=std, a=-2 * std, b=2 * std)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, .1)
            nn.init.trunc_normal_(self.transforms, mean=0., std=.1, a=-.2, b=.2)
            expected = Protocol(profile=profile).describe()["parameters"]
            if sum(p.numel() for p in self.parameters()) != expected:
                raise RuntimeError("Full-width parameter identity mismatch")

        def encode(self, images):
            if images.ndim != 4 or tuple(images.shape[1:]) != (1, 36, 36) or not 1 <= images.shape[0] <= 128:
                raise ValueError("Expected complete B,1,36,36 normalized images")
            if not images.is_floating_point() or not bool(torch.isfinite(images).all()) or bool(((images < 0) | (images > 1)).any()):
                raise ValueError("Expected finite normalized floating-point pixels")
            raw = self.primary_conv(torch.relu(self.conv1(images)))
            if self.profile == "author2017":
                raw = raw / 32  # released PrimaryCaps one-iteration softmax
            raw = raw + self.primary_bias[None, :, None, None]
            # Match released type,row,column,atom flatten order.
            primary = raw.reshape(-1, 32, 8, 10, 10).permute(0, 1, 3, 4, 2).reshape(-1, 3200, 8)
            primary = tensor_squash(primary)
            votes = torch.einsum("bni,njoi->bnjo", primary, self.transforms)
            return tensor_route(votes, self.digit_bias)

        def reconstruct(self, capsules, labels):
            if labels.ndim != 1 or labels.shape[0] != capsules.shape[0] or labels.dtype != torch.int64:
                raise ValueError("One int64 digit label per example is required")
            if bool(((labels < 0) | (labels > 9)).any()):
                raise ValueError("Label outside 0..9")
            mask = torch.nn.functional.one_hot(labels, num_classes=10).to(capsules.dtype)
            return self.decoder((capsules * mask.unsqueeze(-1)).flatten(1))

        def forward(self, images, labels=None):
            capsules = self.encode(images)
            scores = torch.linalg.vector_norm(capsules, dim=-1)
            if labels is None:
                return scores, capsules
            if labels.ndim != 2 or tuple(labels.shape) != (images.shape[0], 2) or bool((labels[:, 0] == labels[:, 1]).any()):
                raise ValueError("Two distinct labels per example are required")
            recon = torch.stack([self.reconstruct(capsules, labels[:, k]) for k in range(2)], dim=1)
            return scores, capsules, recon

    return FullCapsNet()


def paired_loss(scores, labels, predicted, targets, profile):
    import torch
    Protocol(profile=profile)
    batch = scores.shape[0]
    if batch < 1 or tuple(scores.shape) != (batch, 10) or tuple(labels.shape) != (batch, 2):
        raise ValueError("Invalid complete paired classification batch")
    if labels.dtype != torch.int64 or bool(((labels < 0) | (labels > 9)).any()) or bool((labels[:, 0] == labels[:, 1]).any()):
        raise ValueError("Invalid paired labels")
    if tuple(predicted.shape) != (batch, 2, 1296) or targets.shape != predicted.shape:
        raise ValueError("Invalid complete paired reconstructions")
    if any(not bool(torch.isfinite(value).all()) for value in (scores, predicted, targets)):
        raise ValueError("Non-finite loss input")
    if bool((scores < 0).any()) or any(bool(((value < 0) | (value > 1)).any()) for value in (predicted, targets)):
        raise ValueError("Loss inputs outside normalized range")
    positive = torch.nn.functional.one_hot(labels, num_classes=10).sum(dim=1).to(scores.dtype)
    terms = positive * torch.relu(.9 - scores).square() + .5 * (1 - positive) * torch.relu(scores - .1).square()
    classification = terms.sum(dim=1).mean()
    if profile == "author2017":
        classification = classification / 20
    reconstruction = .0005 * (predicted - targets).square().sum(dim=(1, 2)).mean()
    return classification + reconstruction, {"classification": classification, "reconstruction": reconstruction}


def build_optimizer(parameters):
    """Explicit TF-style Adam formula; native parity test remains a required gate."""
    import torch
    from paper_protocol import learning_rate

    class FormulaAdam(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, dict(beta1=.9, beta2=.999, eps=1e-8))
            self.completed_updates = 0

        @torch.no_grad()
        def step(self, closure=None):
            if closure is not None:
                raise ValueError("Closure updates are outside this fixed protocol")
            # Validate every gradient before mutating any optimizer/model state.
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is None or parameter.grad.is_sparse or not bool(torch.isfinite(parameter.grad).all()):
                        raise ValueError("Missing, sparse or non-finite gradient")
            t = self.completed_updates + 1
            for group in self.param_groups:
                b1, b2, eps = group["beta1"], group["beta2"], group["eps"]
                alpha = learning_rate(self.completed_updates) * (1 - b2**t)**.5 / (1 - b1**t)
                for parameter in group["params"]:
                    state = self.state[parameter]
                    if not state:
                        state["m"] = torch.zeros_like(parameter)
                        state["v"] = torch.zeros_like(parameter)
                    state["m"].mul_(b1).add_(parameter.grad, alpha=1 - b1)
                    state["v"].mul_(b2).addcmul_(parameter.grad, parameter.grad, value=1 - b2)
                    parameter.addcdiv_(state["m"], state["v"].sqrt().add_(eps), value=-alpha)
            self.completed_updates = t

        def state_dict(self):
            result = super().state_dict()
            result["completed_updates"] = self.completed_updates
            return result

        def load_state_dict(self, state_dict):
            from paper_protocol import integer
            state_dict = dict(state_dict)
            completed = integer(state_dict.pop("completed_updates"), "completed updates", 0, 2**53)
            super().load_state_dict(state_dict)
            self.completed_updates = completed

    return FormulaAdam(parameters)
