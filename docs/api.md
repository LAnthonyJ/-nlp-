# API and behavior changes

The corrected package lives in `src/nlp_course/`. Files in `coursework/` are historical source snapshots with known defects. Use the corrected package for examples and tests.

## Transformer

`build_transformer(...)` retains the original construction interface. Encoder/decoder blocks own independent parameters. Residual blocks use `x + dropout(sublayer(norm(x)))` (pre-normalization), followed by final stack normalization. This is an explicit architecture correction, not checkpoint-compatible with the original scalar-normalization implementation.

Boolean attention masks use **True = allowed**, with shapes `[B,K]` for padding, `[B,Q,K]` for query/key masks, or `[B,H,Q,K]` for per-head masks. Singleton dimensions can broadcast. `causal_mask(T)` returns `[1,T,T]`; pass it as the target mask. Combine target padding as `causal_mask(T) & padding[:,None,:]`. All-masked rows have zero attention weights and finite gradients (the output projection can still contribute its bias).

`LayerNorm(embed_dim)` now uses feature-wise affine parameters and the standard population-variance formulation. Positional encodings support odd embedding widths and reject sequences exceeding capacity. The model exposes `encode`, `decode`, `project`; it does not implement a combined `forward` method.

## LoRA

`LoRALinear(base_linear, r, alpha, dropout)` freezes the shared base parameters, preserves their device/dtype and initializes the update to zero. Rank and alpha must be positive. `inject_lora_modules` replaces named projections; call `mark_only_lora_as_trainable(model)` to freeze all other model parameters as well.

Call `eval()` before `merge()`. Merge modifies the shared base weight, is idempotent and is inference-only. Its state flag is saved in `state_dict`. Instantiate with the same rank, alpha and other hyperparameters when restoring a checkpoint; call `eval()` after loading a merged state. There is no unmerge API. Save unmerged weights for later training. Do not share a base projection between multiple independently merged adapters.

## Decoding

`DecodingSampler.distribution(logits, strategy=...)` accepts a one-dimensional floating-point vector. Strategies: `greedy`, `topk`, `topp`, `minp`. `sample` accepts an optional `torch.Generator` on the same device as the logits.

Min-P uses `p(token) >= min_p * max_probability`. This intentionally differs from the original absolute cutoff and follows the [Transformers generation definition](https://huggingface.co/docs/transformers/main_classes/text_generation). Top-P includes the first token crossing its cumulative threshold. Top-K is capped at vocabulary size. Tied top-K boundaries use PyTorch's tie ordering; device-independent tie ordering is not promised.

`generate` accepts a model/tokenizer protocol, so the core library does not import Transformers. It handles one prompt, extends its attention mask, applies repetition penalty once per distinct prefix token, stops at EOS, and restores module training flags even on failure. It recomputes the prefix at every step, without a KV cache. Model-based Hugging Face usage requires separately installing its dependencies and obtaining a compatible model; it was not exercised here.

## Evaluation

Run `python -m nlp_course.evaluation INPUT.jsonl --mode gsm8k --output REPORT.json`. Each nonempty input line must contain `prediction` and `reference` strings.

Predictions require `The answer is:`, `####`, or a balanced `\boxed{...}` final-answer marker. References may be plain answers or use those markers. Parsing supports grouped thousands, decimals, scientific notation, rational numbers and simple integer LaTeX fractions; it does not round to integers. Trailing units, percentages, prose, arbitrary algebra and unmarked predictions are not accepted as numeric answers.

`math-strict` compares supported numbers exactly or otherwise compares stripped answer text. It **does not test general symbolic equivalence**, and its scores must not be labeled official MATH benchmark accuracy. For instance `x+x` and `2*x` do not match in this mode. This replaces the missing `util` dependency with an explicitly limited scorer, rather than claiming to recreate that helper.

Invalid numeric predictions count as incorrect and are also reported separately. Malformed references, mismatched counts and malformed JSONL cause explicit errors. Empty evaluations return `accuracy: null`. Per-example records are included. `evaluate_generated` accepts a batch prediction callback and verifies output counts before scoring; no vLLM dependency is required for offline scoring.

## Precision and shared-module repairs — 2026-11-14

Greedy and Top-K choose candidates using original logits, whose ordering is preserved by positive temperature scaling. Filtering and normalization remain in float64 until returning probabilities in the input dtype. Near-equal probabilities can therefore no longer create false ranking ties before selection. Ties already present in the input logits retain the existing tie-ordering policy; rounding of the final returned probabilities is still expected.

LoRA injection enumerates every registered module path. When any alias of a Linear object matches a target name, all its aliases receive the same adapter, including non-target aliases and paths through shared parent modules. The return value counts unique wrapped Linear objects, not alias paths. Repeated injection is a no-op; shared adapter gradients and inference merging apply once per unique module. Aliases must share the same module object: distinct Linear objects merely sharing a weight Parameter are not coalesced.
