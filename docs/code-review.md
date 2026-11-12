# Initial source review

This table records the initial source inspection. Source snapshots are intentionally unchanged. The corrected package and regression tests address these issues as described below.

| Area | Observed behavior | Proposed validation or repair |
| --- | --- | --- |
| Transformer layers | Encoder/decoder stacks repeat the same layer instance with `[layer for _ in range(N)]`. | Test parameter identity; use independent layers if weight sharing is not intended. |
| Attention masks | The same double-unsqueeze is used for different mask shapes. | Test padding and causal masks with explicit shape contracts and future-token isolation. |
| Layer normalization | Scalar affine parameters and sample standard deviation differ from standard feature-wise LayerNorm. | Clarify intended formulation, compare against a reference, and test constant inputs. |
| LoRA construction | Direct construction does not call the custom initialization routine; the injection helper does call it. | Test direct construction and injection separately for zero initial update. The injection path is not subject to the same initialization omission. |
| LoRA merging | Merge changes base weights and zeroes adapter matrices. | Verify evaluation-mode equivalence and idempotence; document that this representation cannot simply be unmerged. |
| Decoding defaults | `generate()` defaults to `greedy`, but dispatch supports only `topp` and `minp`. | Test all advertised strategies and defaults; implement or remove unsupported options. |
| Decoding options | `top_k` and `repetition_penalty` appear in the interface but are not applied. | Align documented behavior with implementation. |
| Min-P naming | Current filter is `probability >= p_min`, an absolute threshold. | Document current semantics; verify the intended algorithm before introducing relative-to-maximum Min-P. |
| GSM8K dependency | Imports `Fraction` from `fraction`, which was not found in the inspected environment. | Check whether standard-library `fractions.Fraction` was intended. |
| Evaluation batching | `batch_data()` uses `(n-1) * batch_size`, causing a negative start when there are fewer samples than one batch. | Test empty, partial and complete batches and preserve every sample exactly once. |
| MATH dependency | Imports `util`; no `util.py` exists beside the assignment scripts. | Locate the original helper and attribution; do not substitute an unvalidated equivalence scorer. |

The LoRA file name mentions training, but the inspected main routine is a model-based smoke demonstration. It is not evidence that a complete fine-tuning experiment has been reproduced.

## Repair status — 2026-11-11

- Transformer: independent layer parameters, explicit boolean masks, finite all-masked attention, feature-wise LayerNorm, pre-normalization residuals and odd-width positional encoding are implemented and tested.
- LoRA: direct initialization, dtype/device matching, frozen base, nonzero adapter gradients, evaluation-only idempotent merging and merged checkpoint restoration are tested. The corrected implementation retains adapter tensors and persists the merge flag.
- Decoding: defaults and all four advertised strategies are implemented; repetition penalty, attention masks, EOS and mode restoration are tested. Min-P now uses a relative threshold; this is an intentional semantic change.
- Evaluation: the corrected package uses standard-library Fraction, complete batching, exact numeric comparisons and explicit invalid-output records. It is independent of vLLM and jsonlines.
- MATH limitation: the original symbolic helper remains unavailable. `math-strict` is an explicitly limited alternative, not a replacement with equivalent benchmark behavior.

See [API](api.md) for compatibility changes and [validation](validation.md) for actual execution evidence. Original coursework paths remain archival; use the package under `src/`.
