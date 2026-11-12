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

## Validation boundary

The original snapshots still match their recorded hashes. The corrected package passed 29 regression tests and offline CPU examples; see validation.md. No full language-model training or benchmark evaluation has been reproduced.

## Repair pass — 2026-11-11

At the owner's request, AI-assisted development introduced corrected modules under `src/nlp_course/`, 29 regression tests, a synthetic CPU training demonstration, offline evaluation, packaging and CI configuration. `transformer.py`, `lora.py` and `decoding.py` revise the concepts and interfaces in the original implementations. `evaluation.py` provides an explicitly limited, dependency-free scoring path rather than recreating the missing MATH helper. These revisions are not attributed as unaided original coursework. The original five files remain unchanged.
