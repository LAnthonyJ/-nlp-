# Qwen2.5-0.5B training and evaluation repair plan

Baseline commit: 52738f8. This historical plan asks whether retraining SFT on the same Base improves held-out accuracy and instruction following, and whether DPO provides additional gains. The numbers below are design targets. The [subsequent report](repair-results/README.md) records the observed results.

The first 420 responses yielded five numerically correct old-SFT answers on 30 math problems and none under strict scoring. Logic and code were both 0/25, while all 25 reference programs passed the same grader. Separate answer format, mathematical correctness and executable code. The actual generation path uses Hugging Face Qwen and PEFT.

## 1. Freeze the baseline and establish an independent evaluation

Add an optional experiment configuration to the capability CLI. Pass model revisions, adapters, templates, terminators, tasks and output directories through generation, scoring, analysis, blind review and plotting. Preserve the original rendering command. Store each run separately. On resumption, verify model, tokenizer, template, task, decoding and protocol hashes and resume after the last complete record. Require one record per expected model and question. Count missing, unscored and truncated outputs. Mark code unscored when Docker is unavailable.

With seed 42, freeze 30 previously unused GSM8K test problems and 25 HumanEval functions. Add 30 logic tasks: ten balanced judgments, ten ordering tasks and ten counts with shuffled facts and programmatic reference answers. Add ten concept and ten writing prompts for blind review. Freeze IDs, references and scoring before checkpoint selection. Prepare separate development tasks from reserved GSM8K train examples, original functions and independent logic templates. Keep the previously examined questions for historical diagnostics.

Evaluate Base, old SFT, old DPO, new SFT and a new DPO candidate only if it qualifies. All course weights use the same English instructions and course template. Report the official same-size Instruct model separately with its native template. Use greedy decoding and a default 256-token budget, validate reference programs in the isolated grader, and open the final set only after locking candidates and configuration.

## 2. Bring data preparation into the repository

Use the training prepare command with a configuration file. Read the two original 1,000-record course datasets. Emit stable IDs, sources, categories and splits for SFT examples and preference pairs. Report missing files, malformed records, identical candidates and empty training targets.

Group duplicate prompts, paraphrases and shared templates before splitting. Use the same seeded group boundaries for SFT and DPO. Check exact and approximate overlap against historical, development and final questions. Record reviewed matches, exclusions and split hashes. Publish counts, sources and examples eligible for redistribution.

Review all 193 suspected preference-generation meta-instructions and check samples outside the rule. Both responses must answer the same request and have a justified preference direction. Record objective validators or human rationale. Quarantine unclear preferences, incorrect SFT answers, incomplete code and contradictory formats. Supplement development categories using disjoint GSM8K train items, solver-backed logic and original functions with tests. Use reviewed responses for open tasks. Record inclusion, exclusion and supplementation counts and pin public dataset revisions and hashes.

## 3. Verify the SFT objective before full training

Move the notebook logic to the training sft command. Share prompt encoding between training and inference. Inspect labels token by token: mask user text, role prefix and padding with -100. Include the assistant completion and one actual end token in the loss. Save the template, special token IDs and adapter. Use the same terminator for inference and exclude incomplete overlength sequences.

Start with the planned LoRA configuration (rank 16, alpha 32) and attention Q/K/V/output projections, microbatch one on a single GPU, gradient accumulation eight, sequence length 512, gradient checkpointing and learning rate 1e-4. Log precision, peak GPU memory, throughput and exclusions. If memory runs out, inspect caches and frozen parameters, bucket by complete sequence length and rerun the smoke test.

Fix 16 training examples with short unique answers. Train for at most 200 optimizer updates. After adapter reload, require at least 14/16 exact greedy answers before proceeding. Investigate label alignment, EOS, gradients, parameter updates and reload equivalence when it fails. This checks the training pipeline.

Compare A, the structurally validated course data, with B, the cleaned and supplemented data. Match initialization, seed, update cap and effective batch size and record supervised-token counts. Evaluate every 50 updates and at the end. Select by the equal-weight mean of strict math accuracy, logic accuracy and code pass rate, breaking ties at the earlier checkpoint. The A/B comparison estimates the combined data change.

## 4. Validate DPO preference direction and the reference model

Start the training dpo command from the selected SFT adapter. Precompute both response log probabilities from its frozen evaluation-mode snapshot before releasing the reference model. Bind the cache to reference weights, examples, encoding and truncation hashes.

Both preference branches must use the same prompt. Accumulate only completion and end-token log probabilities. Check the loss with small tensors, reversed-label gradients, matching initial policy and reference scores, and fixed reference scores after an update. Start at 5e-6, beta 0.1, for at most one epoch. Log preference accuracy, reward margin, repetition and truncation.

Run DPO only if SFT passes the smoke test, beats old SFT on the development objective mean, and reviewed pairs support an independent validation set. To qualify for display, DPO must improve on SFT's mean and lose at most one question on any objective task. Otherwise publish the selected SFT with the observed regressions. These are selection rules.

## 5. Acceptance and reporting

Run CPU regressions and targeted checks for labels and EOS, empty responses, group leakage, preference direction, adapter reload, changed resume configurations and missing records. Distinguish correct numbers with invalid formatting, a correct intermediate number with a wrong final answer, invalid code syntax and executable code that fails tests. Exercise one training step locally and the full path on real GPU weights. Provide runnable preparation, training and reevaluation commands.

For each locked holdout task report counts, correct answers, intervals and paired gains and losses for old SFT → new SFT and new SFT → new DPO. Report strict and numeric math scores together. Break logic down by category and constant baseline. Report code syntax, test outcomes, failure types and truncation. Blind-score concepts and writing or mark them pending. Attribute a format-only gain to format.

Publish the new report alongside the first diagnostic with task and error plots, repaired and still-failing examples, split and weight hashes, software versions and statistics needed to redraw the figures. Document the local assets required for retraining.

The historical plan called for revocation of a previously exposed notebook credential by its owner, environment-based credential loading and clearing saved outputs. It called for inspecting staged notebook content and recording locations only. The initial round uses no paid generation API.
