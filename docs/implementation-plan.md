# Implementation plan — 2026-11-11

The owner requested repairs, tests and runnable examples after reviewing the initial inventory. Execute the existing design in these steps:

1. Preserve `coursework/` and source hashes; introduce corrected modules in `src/nlp_course/`.
2. Repair independent Transformer layers, boolean mask shapes, normalization and residual placement.
3. Repair LoRA direct construction, dtype/device, parameter freezing and inference merge lifecycle.
4. Complete decoding strategies, parameter validation and repetition penalty. Define relative Min-P explicitly.
5. Replace dependency-bound scoring with offline exact numeric and strict-text scoring. Expose a batched generation callback; do not claim general symbolic equivalence.
6. Verify invariants with standard-library unittest and run small CPU examples without model downloads.
7. Update documentation with actual commands, outcomes and remaining limits.

No GPU benchmark, remote publication or new model accuracy claim is part of this repair pass.
