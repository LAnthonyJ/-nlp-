# Validation record

Executed locally on 2026-11-11, on Windows with Python 3.13.7 and PyTorch 2.9.0+cpu. All execution used CPU. No model or dataset downloads were required.

## Installation

A repository-local `.venv` was created with `--system-site-packages` to reuse the installed PyTorch runtime. Editable installation succeeded with `pip install --no-index --no-cache-dir --no-deps --no-build-isolation -e .`.

The sandbox's default temporary path contains braces that trigger an existing setuptools variable-expansion error. Pointing `TEMP` and `TMP` to the ignored repository-local `.build-tmp` directory resolved the build failure. No source modification or global environment change was needed for this workaround. This is not evidence of a fully isolated clean installation from newly downloaded dependencies.

## Tests

Command: `.venv\Scripts\python.exe -m unittest discover -s tests -v`

Result: **29 tests passed**, no failures or skipped tests, including after editable installation.

| Area | Coverage |
| --- | --- |
| Transformer | Independent parameter storage, LayerNorm reference comparison, constant inputs, odd positional width, sequence capacity, causal isolation, padding isolation, 3D/4D masks, all-masked backward, output probabilities and gradients |
| LoRA | Zero initial update, float64 preservation, frozen base, adapter gradients, dropout-disabled merge equivalence, idempotence, checkpoint state, invalid arguments, nested injection and model-wide freezing |
| Decoding | Candidate distributions, Top-P crossing token, relative Min-P boundaries, deterministic seeded sampling, repetition penalty, invalid arguments, default generation, EOS, attention mask growth and training-mode restoration |
| Evaluation | Empty/partial/full batches, exact rational answers, no integer rounding, malformed inputs, nested boxed answers, invalid-output accounting, strict math limitations, backend output counts, CLI report and unchanged snapshot hashes |

## Executed examples

Commands:

```powershell
.\.venv\Scripts\python.exe examples/cpu_demo.py
.\.venv\Scripts\python.exe -m nlp_course.evaluation examples/toy_predictions.jsonl --output outputs/toy-report.json
```

The recorded evaluation was saved under `docs/validation/toy-evaluation.json` using the same CLI with that output path.

- Transformer forward/backward completed with output `[2,5,16]`.
- Synthetic LoRA regression MSE decreased from `0.0459489636` to `0.00000636435` over 80 Adam steps, seed 42.
- Maximum output difference after merging: `1.78813934e-7`.
- Toy JSONL report: four examples, two correct, one invalid output. This file intentionally includes a wrong fractional answer and a missing final answer to illustrate accounting.

Raw records: [CPU example](validation/cpu-demo.json), [toy JSONL scoring](validation/toy-evaluation.json). The synthetic metrics are demonstrations, not GSM8K/MATH or pretrained-model benchmark results.

## Not exercised

- GPU execution, model downloads, real pretrained-model generation and full fine-tuning.
- General symbolic equivalence for MATH; only documented strict scoring is implemented.
- Remote GitHub Actions, which is configured but has not run.
- A clean dependency installation on a new machine or all declared Python versions.
