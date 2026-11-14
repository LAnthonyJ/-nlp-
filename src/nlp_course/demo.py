"""Shared offline demonstration; synthetic data, not language-model benchmarks."""
import platform
import torch
from torch import nn
from nlp_course.transformer import build_transformer, causal_mask
from nlp_course.lora import LoRALinear, count_model_params
from nlp_course.decoding import DecodingSampler
from nlp_course.evaluation import score_predictions


def run_cpu_demo(include_trace=False):
    """Return the original CPU report, optionally with measured plotting traces."""
    torch.manual_seed(42)
    torch.set_num_threads(1)
    model = build_transformer(16, 16, 5, 5, embed_dim=8, N=2, heads=2, hidden_dim=16, dropout=0)
    src, tgt = torch.randint(0, 16, (2, 5)), torch.randint(0, 16, (2, 5))
    encoded = model.encode(src, None)
    log_probs = model.project(model.decode(tgt, encoded, None, causal_mask(5)))
    loss = nn.functional.nll_loss(log_probs.reshape(-1, 16), tgt.reshape(-1))
    loss.backward()

    adapter = LoRALinear(nn.Linear(4, 3), r=2, alpha=2)
    x = torch.randn(64, 4)
    with torch.no_grad():
        target = adapter(x) + x[:, :1] @ torch.tensor([[0.3, -0.2, 0.1]])
    optimizer = torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad], lr=0.03)
    before = nn.functional.mse_loss(adapter(x), target).item()
    history = [{"step": 0, "mse": before}] if include_trace else None
    for step in range(1, 81):
        optimizer.zero_grad()
        adapter_loss = nn.functional.mse_loss(adapter(x), target)
        adapter_loss.backward()
        optimizer.step()
        if include_trace:
            with torch.no_grad():
                history.append({"step": step, "mse": nn.functional.mse_loss(adapter(x), target).item()})
    adapter.eval()
    after = nn.functional.mse_loss(adapter(x), target).item()
    expected = adapter(x).detach()
    adapter.merge()
    merge_error = (adapter(x) - expected).abs().max().item()
    assert after < before
    torch.testing.assert_close(adapter(x), expected)
    logits = torch.tensor([0.6, 0.3, 0.1]).log()
    sampler = DecodingSampler()
    distributions = {s: sampler.distribution(logits, strategy=s, top_k=2, top_p=0.8, min_p=0.5).tolist()
                     for s in ("greedy", "topk", "topp", "minp")}
    report = dict(seed=42, device="cpu", torch_version=torch.__version__,
                  transformer_shape=list(log_probs.shape), transformer_loss=loss.item(),
                  lora_synthetic_loss_before=before, lora_synthetic_loss_after=after,
                  lora_merge_max_error=merge_error, lora_parameter_counts=count_model_params(adapter),
                  sampling_distributions=distributions,
                  toy_evaluation=score_predictions(["The answer is: 1/2", "no answer"], ["0.5", "2"]))
    if include_trace:
        report['trace'] = dict(
            loss_history=history,
            merge_outputs=dict(before=expected.tolist(), after=adapter(x).detach().tolist()),
            sampling_logits=logits.tolist(),
            sampling_settings=dict(temperature=1.0, top_k=2, top_p=0.8, min_p=0.5),
            original_distribution=torch.softmax(logits, dim=0).tolist(),
            training=dict(seed=42, steps=80, learning_rate=0.03, rank=2, alpha=2,
                          optimizer='Adam', task='Synthetic linear regression', samples=64),
            environment=dict(python=platform.python_version(), platform=platform.platform(),
                             torch=torch.__version__, device='cpu'))
    return report
