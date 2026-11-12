# NLP portfolio repository design

Date: 2026-11-11

## Goal

Build one coherent NLP coursework implementation repository within a broader collection of worthwhile projects. Support CS/AI master's applications and employment through explainable code, verified behavior and honest contribution records.

## Scope and structure

The first phase imports five existing Python scripts with immutable provenance. `coursework/` is the source snapshot; `docs/` explains origin and limitations. English is the main public reading language, with a Chinese companion README.

The repair phase introduced corrected reusable implementations under `src/nlp_course/`, tiny CPU examples under `examples/`, and invariant-based tests under `tests/`. Source snapshots remain available for comparison. BPE and final-project SFT/DPO require a separate content review before inclusion; training checkpoints are not Git source files.

## Behavior and data flow

- Transformer: token IDs and explicit padding/causal masks -> encoder/decoder hidden states -> vocabulary scores.
- LoRA: base linear layer -> frozen base plus trainable low-rank update -> optional inference merge.
- Decoding: logits -> temperature and candidate filtering -> normalized probabilities -> sampled token.
- Evaluation: model outputs and reference answers -> documented extraction -> score plus per-example records.

Pure numerical components should work without model downloads. Missing optional model dependencies should produce actionable messages. Invalid ranks, mask shapes and sampling parameters should be rejected explicitly. Evaluation failures should be recorded separately from model accuracy.

## Repair acceptance criteria

1. Independent Transformer layers and correct causal-mask behavior are tested on tiny tensors.
2. LoRA starts with zero update; only adapters receive gradients; merging preserves evaluation outputs.
3. All advertised decoding strategies work, including the default; threshold boundaries are tested.
4. Evaluation batch coverage and answer extraction are checked using hand-verifiable examples.
5. A tested dependency specification and CPU example are documented.
6. Large-model results are included only when actual configurations, outputs and execution evidence are available.

## Publication and resume readiness

The owner confirmed on 2026-11-11 that all five included coursework implementations were completed by them. Attribution is recorded in provenance.md. Track what has actually been run, distinguish imported historical results from newly reproduced experiments, and record AI-assisted repository work separately.

The current phase does not require cloud training, model downloads or a public remote. Publication preparation on 2026-11-13 uses the connected GitHub account LAnthonyJ and its ID-based noreply commit email.

Implementation status: the CPU repair scope is complete; see validation.md. Real-model experiments remain outside this pass.
