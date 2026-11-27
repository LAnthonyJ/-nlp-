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
- A clean dependency installation on a new machine or all declared Python versions.

## Publication validation — 2026-11-13

The initial commit `c19b40650c874ec02965123c4b33472a00d68475` was pushed to [LAnthonyJ/small-language-model-for-education](https://github.com/LAnthonyJ/small-language-model-for-education). [GitHub Actions CPU tests](https://github.com/LAnthonyJ/small-language-model-for-education/actions/runs/43263272641) completed successfully on Ubuntu / Python 3.11, including dependency installation, unittest, the CPU demonstration and JSONL scoring. The Git index was also checked against all five original coursework SHA-256 hashes.

## P2 review regressions — 2026-11-13

Reviewed baseline: `1c513ae`. Added nine regression tests in `tests/test_review_regressions.py` before changing the implementation. The baseline reproduced nested-box extraction errors, NaN sampling distributions at small temperatures, and silent attention input reshaping. New assertions failed on the baseline; the complete repaired suite passes **38 tests** on Windows / Python 3.13.7 / PyTorch 2.9.0+cpu.

- Final-answer extraction scans complete outer boxed spans and skips markers within those spans. Unclosed outer boxes return no answer; later independent final-answer markers still take precedence. Nested expressions are retained, not reduced to an inner numeric answer.
- Sampling probabilities use widened arithmetic and a shift-invariant temperature calculation. Small temperatures center logits before division; large temperatures divide before centering to avoid subtraction overflow. Result probabilities retain the input dtype. Tests include float16/float32 overflow, a tiny positive temperature and opposite float64 extremes, across all four strategies.
- Attention validates rank, batch, feature width and key/value sequence lengths before projection or reshaping. Different query/key sequence lengths remain valid for cross-attention.

Commands: `python -m unittest discover -s tests -v` and `python examples/cpu_demo.py`. The prior 29 tests, including source hashes, causal isolation and LoRA merge checks, remain part of the suite. No GPU validation was performed for these numerical edge cases.

## Precision and alias review regressions — 2026-11-14

Reviewed baseline: `20bfd8b`. Six new tests in `tests/test_precision_and_aliases.py` reproduced false Greedy/Top-K ranking ties and broken shared-module aliases before implementation changes. The repaired complete suite passes **44 tests locally**, including all previous overflow, boxed-answer, attention-shape and LoRA merge regressions.

Ranking tests exercise float16, bfloat16, float32 and float64, with close logits and temperatures up to `1e300`. They verify token support against raw-logit order, probability values against a widened reference, and preservation of output dtype. Shared-module tests cover two target aliases, a non-target alias registered first or last, shared parents, zero-update equivalence, adapter gradients, unique parameter counts, one-time merging and repeated injection.

The checks ran on Windows / Python 3.13.7 / PyTorch 2.9.0+cpu with `python -m unittest discover -s tests -q`. GPU execution was not tested in this pass.

## Static visualization validation — 2026-11-14

Added a shared CPU demo runner and separate collect/render commands. Local Windows / Python 3.13.7 / PyTorch 2.9.0+cpu / Matplotlib 3.10.7 execution passed **52 tests** with the plotting dependency installed. The default example remains JSON-only with its original fields. The renderer produced five SVG and five PNG files; all five PNGs were visually inspected for clipping, label overlap and legibility.

Recorded data contains 81 actual loss measurements (steps 0–80), 64×3 paired merge outputs and explicit decoding settings. The four-example evaluation report remains distinct from the demonstration's two-example report. Source snapshots still pass SHA-256 checks. See [visualization reproduction and data links](visualization.md).

## Real-checkpoint diagnostic validation — 2026-11-15

The complete local CPU suite passed **64 tests** on Windows / Python 3.13.7 / PyTorch 2.9.0+cpu with Matplotlib installed. The added capability checks cover frozen task preparation and leakage screening, separate format/numeric failures, bounded numeric parsing, code completion markers, bounded UTF-8 subprocess output and deadlines, unavailable execution accounting, blind-review protection, incomplete rating rejection, repeat exclusion, stage regressions and static rendering. The existing CPU demonstration and original coursework hashes remain covered.

In the separate CUDA environment, PyTorch 2.9.0+cu128 / Transformers 5.9.0 / PEFT 0.21.3 generated 10 development responses and 140 formal responses for each of Base, SFT and DPO: **30 development + 420 formal outputs**. Each stage passed an eight-token custom/reference greedy agreement check. The generation function hash stayed unchanged throughout formal runs, and resumable configurations record actual weight and source hashes.

All 420 formal responses were scored. The selected 25 canonical HumanEval programs passed the same bounded, network-isolated Python 3.11 Docker harness; a Unicode/no-trailing-newline fixture also passed. All 1,319 converted GSM8K pairs matched the official data. Published [results](capability-results/README.md) separate unchanged strict protocol scores from post-hoc numeric-format matching. These checks validate this diagnostic pipeline, not general model quality or absence of benchmark contamination.

Both real-checkpoint PNG figures were visually inspected, with zero outcomes, sample counts and metric names explicit; matching SVG files are provided. Source and artifact hashes, per-input scores and aggregate analyses are retained in the public results directory. Raw open-task answers, local paths, training samples, model weights and review keys remain local. Human concept/writing quality is still pending: the 66-row blind sheet has not been treated as completed ratings. No new training, full-benchmark run or multi-seed causal comparison was performed.
# Training repair validation

2026-11-18. The local CPU regression suite passed all 70 tests. Real GPU validation completed 525 held-out responses, a saved-adapter reload check with 16/16 complete answers, and all 25 selected canonical HumanEval programs passing the isolated scorer. The SFT A/B runs each used 50 optimizer steps; a 28-step DPO continuation did not improve the development objective mean. Open-task human ratings and the original coursework content review remain pending. See the [measured repair report](repair-results/README.md) and [runtime fingerprints](repair-results/runtime.json).
