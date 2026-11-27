# Small Language Model for Education

[![CPU tests](https://github.com/LAnthonyJ/small-language-model-for-education/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/LAnthonyJ/small-language-model-for-education/actions/workflows/tests.yml)

This project grew out of five NLP coursework implementations. You can inspect a causal attention mask on a CPU, train and merge a low-rank adapter, or follow a controlled experiment that tests whether direct preference optimization (DPO) improves a supervised Qwen2.5-0.5B checkpoint.

## Can a future token change an earlier output?

After [installing the package](#run-locally), run this example in Python. It changes only the last token in a four-token sequence and compares the first three attention outputs.

```python
import torch
from nlp_course.transformer import MultiHeadAttention, causal_mask

torch.manual_seed(42)
attention = MultiHeadAttention(embed_dim=8, num_heads=2, dropout=0).eval()
tokens = torch.randn(1, 4, 8)  # one sequence, four tokens, eight features
changed = tokens.clone()
changed[:, -1] += 10

with torch.no_grad():
    mask = causal_mask(4)
    before = attention(tokens, tokens, tokens, mask)
    after = attention(changed, changed, changed, mask)

torch.testing.assert_close(before[:, :3], after[:, :3])
print((before[:, :3] - after[:, :3]).abs().max().item())  # 0.0
```

Try replacing `mask = causal_mask(4)` with `mask = None` and rerun the example. The assertion fails because earlier positions can now attend to the changed token. [Attention implementation](src/nlp_course/transformer.py) · [Mask regression tests](tests/test_components.py)

## Does DPO improve a small language model?

Starting from the same supervised fine-tuning (SFT) checkpoint, I compared two updates. Standard DPO trained on 511 pairs of correct answers and model errors. A supervised control saw the same prompts and only the correct answers. Both branches had the same learning-rate grid and maximum update budget. Checkpoints were selected on development data before opening the holdout.

| Four-task mean accuracy | Starting SFT | Standard DPO | Correct-answer-only control | DPO gain over SFT, paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| New holdout · 768 prompts | 44.66% | 45.57% | **52.73%** | +0.91 pp [+0.13, +1.82] |
| Public suite · 13,680 prompts | 43.49% | 43.64% | Not evaluated | +0.15 pp [−0.35, +0.66] |

DPO answered seven more holdout prompts correctly than SFT. That fell one prompt short of the prespecified one-percentage-point target. On the public suite, the interval included zero. The strongest holdout result came from the supervised control. [Experiment report](docs/hard-pairs-v3-results/README.md) · [Scoring data](docs/hard-pairs-v3-results/metrics.json) · [Training code](src/nlp_course/hard_pairs.py)

An earlier audit explains why the tokenization details matter. Splitting a jointly encoded prompt and answer at the separately measured prompt length could drop the answer's first character. Replaying the repaired encoding restored 643 instruction demonstrations and 136 preferred answers. [Audit implementation](src/nlp_course/progression_audit.py) · [Boundary tests](tests/test_completion_v2.py) · [Repair experiment](docs/progression-v2-results/README.md)

## Read the implementations

| Question | Source | Evidence |
| --- | --- | --- |
| How do causal and padding masks affect attention? | [Transformer](src/nlp_course/transformer.py) | [Mask, gradient, and layer-independence tests](tests/test_components.py) |
| Can a trained LoRA update be merged into the base weight? | [LoRA layers and injection](src/nlp_course/lora.py) | [Measured merge error](docs/figures/lora-merge.png) · [Shared-module tests](tests/test_precision_and_aliases.py) |
| Which tokens survive top-k, top-p, or relative min-p sampling? | [Decoding sampler](src/nlp_course/decoding.py) | [Candidate distributions](docs/figures/decoding.png) · [API and examples](docs/api.md#decoding) |
| How are wrong or unparseable answers counted? | [Answer scoring](src/nlp_course/evaluation.py) | [Input fixture](examples/toy_predictions.jsonl) · [Per-answer results](docs/validation/toy-evaluation.json) |

## Run locally

From the repository root, use a Python 3.10+ virtual environment. These commands install the tested CPU build of PyTorch and run the offline examples.

```sh
python -m pip install -r requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
python examples/cpu_demo.py
python -m nlp_course.evaluation examples/toy_predictions.jsonl --output outputs/toy-report.json
python -m unittest discover -s tests -v
```

The CPU demo runs one Transformer backward pass and 80 LoRA optimizer steps on synthetic data. The four-answer scoring fixture produces two correct answers and one invalid output. The Qwen experiments use separate GPU runs and archived training inputs. [CPU demo source](src/nlp_course/demo.py) · [Recorded output](docs/validation/cpu-demo.json) · [GPU reproduction guide](docs/hard-pairs-v3.md)

## Project history

The [coursework archive](coursework/) contains the five original submissions. The maintained package, tests, and reports were developed with AI assistance. Earlier experiments document how code and math scoring were repaired and how six Qwen checkpoints compared on public small-model benchmarks. [Provenance](docs/provenance.md) · [Scoring repairs](docs/repair-results/README.md) · [Public benchmark study](docs/smollm2-results/README.md)
