# Provenance and contributions

The source manifest maps each copied file to its original path relative to the local course workspace. SHA-256 values identify the exact imported versions.

## Coursework implementation

On 2026-11-11, the repository owner confirmed that they completed all five included coursework implementations:

- Transformer components (`coursework/transformer/model.py`).
- LoRA layers, module injection and parameter utilities (`coursework/lora/lora.py`).
- Decoding implementations (`coursework/decoding/sampling.py`).
- GSM8K evaluation code (`coursework/evaluation/eval_gsm8k.py`).
- MATH evaluation code (`coursework/evaluation/eval_math.py`).

This attribution records the owner's confirmation. It describes implementation contributions, not invention of the underlying algorithms or ownership of external models, libraries and datasets. Assignment-style comments remain in the unchanged source snapshots.

## Repository preparation

The original files remain untouched. Repository organization, source inventory, initial review notes and documentation were prepared with AI assistance. No algorithm implementation was changed during the import or attribution update. Subsequent implementation changes are recorded below.

## External material and licensing

The vendored subword-nmt project, model checkpoints and datasets were not copied into this repository. External libraries, models and datasets retain their respective attribution and license terms. A license for this repository has not yet been selected.

## Initial validation boundary

At the initial repair, the original snapshots matched their recorded hashes and the corrected package passed 29 regression tests and offline CPU examples; see validation.md. Later validation is recorded chronologically below. No full language-model training was reproduced.

## Repair pass — 2026-11-11

At the owner's request, AI-assisted development introduced corrected modules under `src/nlp_course/`, 29 regression tests, a synthetic CPU training demonstration, offline evaluation, packaging and CI configuration. `transformer.py`, `lora.py` and `decoding.py` revise the concepts and interfaces in the original implementations. `evaluation.py` provides an explicitly limited, dependency-free scoring path rather than recreating the missing MATH helper. These revisions are not attributed as unaided original coursework. The original five files remain unchanged.

## Static visualizations — 2026-11-14

At the owner's request, AI-assisted development extracted the existing CPU demonstration into a reusable runner, instrumented actual measurements, added static plotting and regression tests, and updated bilingual documentation and CI. The learning curve is a single fixed-seed synthetic training run. All figures are generated from recorded data, not illustrative model-performance estimates. Original coursework and historical validation JSON files are preserved.

## Real-checkpoint capability workflow — 2026-11-15

At the owner's request, AI-assisted development added independent GPU checkpoint evaluation, fixed main/paired task preparation, isolated code execution, anonymous human-review export/import, statistics and bilingual reports. Existing Qwen-based SFT/DPO adapters are original course training artifacts; no new model training was performed in this evaluation. Their task results are separate evidence from the custom LoRALinear component tests. Human open-task scores remain pending until the owner supplies the completed blind-review sheet. Machine-specific input paths are configured locally under ignored outputs/.

The first execution completed 420 formal responses, 30 development outputs and isolated objective scoring. Published aggregates, objective excerpts, plots and provenance are in [capability-results](capability-results/README.md). Original weights, training samples, full raw outputs and the private review key remain local. A post-hoc numeric-format analysis is explicitly separated from the unchanged strict scores. The report records failed tasks and limitations; it makes no claim of state-of-the-art performance or a general causal DPO benefit.
