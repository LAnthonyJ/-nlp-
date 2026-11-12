"""LoRA linear adapters; derived from coursework/lora/lora.py.

Merging is an irreversible inference operation on the shared base layer.
Save an unmerged checkpoint if training is to be resumed.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base_linear, r=8, alpha=16, dropout=0.0):
        super().__init__()
        if not isinstance(base_linear, nn.Linear):
            raise TypeError("base_linear must be nn.Linear")
        if isinstance(r, bool) or not isinstance(r, int) or r < 1:
            raise ValueError("r must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.in_features, self.out_features = base_linear.in_features, base_linear.out_features
        self.r, self.alpha, self.scaling = r, alpha, alpha / r
        self.base_weight = base_linear.weight
        self.base_bias = base_linear.bias
        self.base_weight.requires_grad_(False)
        if self.base_bias is not None:
            self.base_bias.requires_grad_(False)
        kwargs = dict(device=self.base_weight.device, dtype=self.base_weight.dtype)
        self.lora_A = nn.Linear(self.in_features, r, bias=False, **kwargs)
        self.lora_B = nn.Linear(r, self.out_features, bias=False, **kwargs)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("_merged", torch.tensor(False, device=self.base_weight.device))
        self.reset_parameters()

    @property
    def merged(self):
        return bool(self._merged.item())

    def reset_parameters(self):
        if self.merged:
            raise RuntimeError("Cannot reset a merged adapter")
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def train(self, mode=True):
        if mode and self.merged:
            raise RuntimeError("Merged adapters are inference-only")
        return super().train(mode)

    @torch.no_grad()
    def merge(self):
        if self.merged:
            return
        if self.training:
            raise RuntimeError("Call eval() before merging")
        self.base_weight.add_((self.lora_B.weight @ self.lora_A.weight) * self.scaling)
        self._merged.fill_(True)

    def forward(self, x):
        if self.merged and self.training:
            raise RuntimeError("Call eval() after loading a merged checkpoint")
        y = F.linear(x, self.base_weight, self.base_bias)
        if not self.merged:
            y = y + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return y


TARGET_LINEAR_KEYWORDS = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
}


def inject_lora_modules(model, r=8, alpha=16, dropout=0.0):
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_LINEAR_KEYWORDS]
    for name, module in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        adapter = LoRALinear(module, r, alpha, dropout)
        adapter.train(module.training)
        setattr(parent, child, adapter)
    return len(targets)


def mark_only_lora_as_trainable(model):
    adapters = [m for m in model.modules() if isinstance(m, LoRALinear)]
    if any(m.merged for m in adapters):
        raise RuntimeError("Cannot train merged adapters")
    for p in model.parameters():
        p.requires_grad_(False)
    for m in adapters:
        m.lora_A.weight.requires_grad_(True)
        m.lora_B.weight.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total, trainable / total if total else 0.0
