# NLP Course Implementations

A collection of NLP coursework implementations with corrected reusable components, regression tests and offline CPU examples.

[GitHub repository](https://github.com/LAnthonyJ/-nlp-) · [中文说明](README.zh-CN.md) · [API and behavior changes](docs/api.md) · [Validation](docs/validation.md) · [Provenance](docs/provenance.md)

## What this project demonstrates

| Area | Corrected implementation | Verified behavior |
| --- | --- | --- |
| Transformer | [transformer.py](src/nlp_course/transformer.py) | Independent layers, causal/padding masks, feature-wise normalization, backward pass |
| LoRA | [lora.py](src/nlp_course/lora.py) | Zero initial update, frozen base, adapter gradients, merge equivalence and checkpoint state |
| Decoding | [decoding.py](src/nlp_course/decoding.py) | Greedy, Top-K, Top-P, relative Min-P, repetition penalty and EOS handling |
| Evaluation | [evaluation.py](src/nlp_course/evaluation.py) | Exact numeric scoring, explicit invalid outputs, safe batch coverage, JSONL reports |

`src/nlp_course/` is the corrected package. `coursework/` preserves the five original files byte-for-byte for provenance and comparison; those snapshots still contain the defects documented in the [initial review](docs/code-review.md).

## Quick start

Python 3.10+ is declared supported; local execution was verified on Windows, Python 3.13.7 and PyTorch 2.9.0+cpu. A Linux/Python 3.11 GitHub Actions workflow is configured but has not run remotely.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.txt --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
python examples/cpu_demo.py
python -m nlp_course.evaluation examples/toy_predictions.jsonl --output outputs/toy-report.json
```

On macOS/Linux, activate with `source .venv/bin/activate` and install a PyTorch build suitable for your platform before installing the package. Platform-specific installation outside the tested Windows environment is not yet verified. If PowerShell script activation is unavailable, invoke `.\.venv\Scripts\python.exe` directly in place of `python`.

The CPU demo and tests do not download models or datasets. The demo performs a tiny Transformer forward/backward pass, optimizes a LoRA adapter on synthetic data, verifies merge equivalence, and prints sampling distributions and toy evaluation results.

## Recorded validation

- **29 tests passed** with standard-library `unittest`.
- Transformer example output shape: `[2, 5, 16]` with finite gradients.
- Synthetic LoRA MSE: `0.045949` to `0.000006364` over 80 optimization steps (seed 42).
- Maximum adapter merge difference: approximately `1.79e-7`.
- Both the CPU demo and JSONL evaluation CLI completed successfully after editable installation.

These measurements demonstrate correctness on small synthetic examples, not language-model quality or published benchmark accuracy. Raw outputs: [CPU demo](docs/validation/cpu-demo.json), [toy evaluation](docs/validation/toy-evaluation.json).

## Evaluation scope

The evaluation CLI scores saved predictions; it does not require vLLM or Transformers. Predictions must contain a supported final-answer marker. `gsm8k` mode compares exact numeric values without rounding. `math-strict` accepts supported numeric equality or stripped text equality, **not general symbolic equivalence or official MATH scoring**. Use `evaluate_generated` to integrate a generation backend that returns one string per prompt.

See [API documentation](docs/api.md) for parsing rules, mask conventions and intentional changes from the coursework.

## Repository layout

```text
src/nlp_course/       Corrected library
coursework/          Unchanged original implementations
examples/            Offline CPU demonstration and toy prediction data
tests/               Regression tests
docs/               Design, provenance, review and validation records
.github/workflows/   CPU CI configuration
```

## Attribution and next steps

The owner confirmed completing all five original coursework implementations. Repository preparation, repairs and regression tests were developed with AI assistance; [provenance](docs/provenance.md) separates these stages. The algorithms and external libraries retain their original attribution. A repository license has not yet been selected.

Future work includes reproducing real model experiments and reviewing SFT/DPO as a separate portfolio project. No large-model training, remote benchmark run or model-performance improvement is claimed here.
