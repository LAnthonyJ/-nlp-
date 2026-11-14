# Real-checkpoint capability evaluation / 真实模型多能力诊断

This workflow compares Qwen2.5-0.5B Base, the existing SFT adapter, and the final DPO adapter. The DPO adapter is a continuation of SFT: load each adapter separately on the base, never stack SFT and DPO adapters. This is an English quick diagnostic, not an exhaustive benchmark or a claim of new training.

## Environment and inputs

Keep the existing CPU environment. Use a separate `.venv-eval` with CUDA PyTorch appropriate for the local GPU and install `.[eval,viz]`. The local machine has an RTX 3060 Laptop / 6GB. A smoke run must verify actual CUDA availability, loading, EOS, user/assistant prompt format and custom/reference greedy decoding before formal generation. FP16, one model at a time and batch size one limit memory demand. Weights are loaded from the existing Hugging Face cache and local `q2_sft_output` / `q2_dpo_output`, without modifying the original files.

Run commands from the repository root. Supply `--train-dir` and `--gsm-data` on another machine, or put those paths in the ignored local file `outputs/capability/local-inputs.json`. Machine-specific paths are not part of the published source. Checkpoints, source data, raw outputs, private review keys and the GPU environment are not committed to Git. Reports and plots intended for publication must be exported explicitly after inspection.

## Commands

```powershell
python -m nlp_course.capability prepare
python -m nlp_course.capability generate --stage base --phase smoke
python -m nlp_course.capability generate --stage sft --phase smoke
python -m nlp_course.capability generate --stage dpo --phase smoke
python -m nlp_course.capability generate --stage base
python -m nlp_course.capability generate --stage sft
python -m nlp_course.capability generate --stage dpo
python -m nlp_course.capability score --execute-code
python -m nlp_course.capability review
python -m nlp_course.capability analyze
python -m nlp_course.capability render
```

The default artifact root is `outputs/capability/`, with a frozen `suite/suite.json` and `run/`. Each generation result is appended per question so the same command can resume. Model/adapter hashes, suite hash and configuration must match existing output; changed configurations require a new run directory. Missing answers stay missing, rather than becoming wrong answers. Generation uses a single shared user/assistant ChatML format, greedy decoding and 256 output tokens; the code budget intervention additionally uses 512. No test-set tuning is allowed.

`prepare --humaneval-data PATH` can use an already downloaded official JSONL or gzipped file. Otherwise it downloads the official HumanEval data. The local GSM8K file is the 1,319-row test split converted to query/response fields. Sources are identified by hashes; exact training-prompt overlaps block preparation, near-overlap candidates block generation pending review. Case/whitespace normalization and word-set similarity are screening tools, not proof of no semantic contamination.

## Suite and scoring

The main suite has **100 questions**: 30 GSM8K questions, 25 generated logical-oracle tasks, 25 HumanEval functions, 10 concept explanations and 10 writing tasks. Seed 42 fixes sampling. These task types and metrics remain separate.

There are another **40 inputs forming 20 controlled pairs**: number magnitude, logical chain length, provided reference facts, one versus three writing constraints, and code output budgets. These are separate diagnostic probes and are not pooled into main-task accuracy. In total each model receives 140 formal/diagnostic prompts; the three stages produce 420 responses, plus 30 development outputs.

- Math uses exact rational final-answer comparison; extraction failure and wrong numeric answers have separate statuses.
- Logic uses exact requested answer formats; a format failure is not automatically evidence of invalid reasoning.
- Code uses the selected HumanEval tests. The official Python 3.11 slim container has no network, no host directory mounts, a read-only root, non-root user, process/memory/CPU limits and a 15-second timeout. Prepare `python:3.11-slim` before scoring. Docker failures remain unavailable, not incorrect. Do not run generated code directly on the host.
- Concept/writing main tasks remain **pending human review**. Reference-context diagnostic keyword coverage is only lexical coverage, not factual accuracy. Writing diagnostic format constraints are automatically checked.
- Objective task pass rates include sample size and Wilson intervals. Small selected samples do not establish broad population capability. A truncated response can still contain a correct answer; truncation is a separate recorded attribute.

## Anonymous scoring / 匿名人工评分

After all three formal runs, `review` exports `blind-review.csv`: 60 open-task answers plus six repeated answers, randomly ordered and stripped of stage labels. Keep `private-review-key.json` away from the reviewer. This is identity blinding, not a guarantee that answer style cannot suggest a model stage.

Score each dimension from **1 to 5**:

| Score | Accuracy / task completion | Completeness / coherence | Clarity / expression |
|---|---|---|---|
| 1 | Mostly false or fails the request | Essential content missing or incoherent | Difficult to understand |
| 2 | Major errors or requirements missed | Several major gaps | Frequent ambiguous/awkward passages |
| 3 | Main point broadly correct, with a meaningful error/gap | Mostly connected but incomplete | Understandable with noticeable issues |
| 4 | Correct and nearly complete | Covers key points coherently | Clear, minor defects only |
| 5 | Correct, satisfies all material requirements | Complete and well organized | Precise and fluent |

For concept questions, use the provided facts to score factual accuracy, completeness and clarity. For writing, evaluate completion, coherence and expression. Record explanations in `notes`, not only numbers. A single reviewer is explicitly identified as a limitation.

```powershell
python -m nlp_course.capability import-review --ratings outputs/capability/run/blind-review.csv
python -m nlp_course.capability analyze
python -m nlp_course.capability render
```

Import rejects unknown/duplicate IDs, missing ratings and values outside 1–5. Repeat answers assess scoring consistency and are excluded from the capability means. The unfilled template is not a completed rating file.

## Evidence and interpretation

`analysis.json` records per-category results, stage wins/regressions/ties, controlled-pair results, human-review status and rating-repeat discrepancy. `report.md` and `cases.json` expose failures selected by stable task ID, not flattering hand-picked examples. Static plots include capability comparisons and boundary probes; human plots appear only after ratings are imported.

Investigate errors at three levels: observations (wrong answer/truncation), interventions (changed context/budget/constraint), and hypotheses (capacity, data quality or preference training). A stage difference with one checkpoint per stage is not universal causal evidence. Inspect the fixed 20-record SFT and 20-record DPO audit samples for factual quality, preference direction, coverage and accidentally retained data-construction instructions. Do not silently repair/retrain the model as part of this evaluation.

The model card's eventual boundary section must say what was measured, identify conditions causing observable failures, and explicitly list Chinese, multi-turn conversation, long-context use and full-benchmark generalization as not evaluated. Report regressions as well as improvements. Human ratings and failure interpretation require the owner's participation; unavailable evidence is never filled with estimated scores.

## Local execution notes (2026-11-15)

The independent GPU environment uses PyTorch 2.9.0+cu128, Transformers 5.9.0 and PEFT 0.21.3. Docker 29.4.0 and the official Python 3.11 slim image passed a sandbox fixture. Initial smoke outputs exposed literal copying of a NUMBER placeholder; formal math prompts instead request the final numeric answer without that placeholder. Both end-of-text and end-of-message tokens stop formal generation. These settings were fixed before the formal objective runs.

The current run directory is `outputs/capability/final-run` (pass `--run-dir` for score/review/analyze/render/import-review commands). Formal generation may first use `--open-only` to prepare blind review; rerun without that flag to resume the remaining questions under the same configuration.

A structural audit found 193/1000 DPO prompts matching a pattern for data-construction instructions, and one duplicate SFT prompt. These are audit candidates, not automatically invalid examples or proof of the cause of any model failure. The fixed audit samples and complete structural counts are retained locally. Main conceptual questions test familiar training themes; lexical overlap checks do not establish that their semantics are wholly unseen.

### Secondary format-sensitivity analysis

During the first SFT objective run, many final-answer phrases omitted the requested colon. Strict numeric scores remain unchanged. A separately labeled post-hoc diagnostic accepts an explicit final-answer phrase with an optional colon/currency symbol; it never takes an arbitrary last number. It is applied consistently to all stages, is not a replacement benchmark score, and does not demonstrate correct reasoning merely by matching a number. Raw strict and relaxed outcomes are both retained for audit.

### Recreate the GPU environment on this Windows machine

```powershell
python -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install 'torch==2.9.0+cu128' --index-url https://download.pytorch.org/whl/cu128
.\.venv-eval\Scripts\python.exe -m pip install -e '.[eval,viz]'
.\.venv-eval\Scripts\python.exe -c "import torch; assert torch.cuda.is_available()"
```

The executed environment reused existing system packages while overriding Torch inside `.venv-eval`; the commands above create a clean environment instead. Actual executed versions are recorded in `outputs/capability/evaluation-environment.json`. The original CPU environment remains separate. Standard GSM8K `question`/`answer` fields and the course's `query`/`response` conversion are both accepted. Keep the original test split and source attribution when obtaining the dataset.

The review exporter refuses to overwrite an existing CSV, to protect ratings already being edited. Use the existing file for import, or deliberately use a new run directory for a separate assessment.
