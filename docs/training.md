# Local repair experiments

Training uses the Hugging Face Qwen2.5-0.5B checkpoint and PEFT adapters. The coursework Transformer is tested independently. GPU training versions used for validation are pinned in `requirements-training.txt`; install these in a CUDA environment. CPU regression tests use the existing CPU environment. Download the model snapshots before invoking training; training itself uses local files only.

## Prepare and freeze

Copy the two original coursework JSON files to `data/raw/coursework`, or edit `configs/repair-prepare.json` with their actual directory. Put the official GSM8K train/test JSONL files and compressed HumanEval file at the paths specified by `configs/repair-experiment.json`. The old public suite index supplies the historical benchmark IDs to exclude. Fresh directories are required when changing data or experiment configurations.

```sh
python -m nlp_course.capability freeze --experiment-config configs/repair-experiment.json
python -m nlp_course.training prepare --config configs/repair-prepare.json
python -m nlp_course.training sft --config configs/repair-overfit.json
```

Preparation creates immutable splits, an audit queue with original file indices and content IDs, and a source/hash manifest. Unreviewed coursework completions are quarantined. A JSON file supplied through the optional `reviews` configuration maps each content ID to `{"approved": true, "reason": "specific verification evidence"}`. Suspected leakage remains excluded even with approval. The optional `structural_sft_control` setting admits structurally valid SFT coursework only for the explicitly labeled control experiment. This control has no content-correctness guarantee.

Programmatic arithmetic, entailment, and arithmetic-function samples have deterministic correct responses. Their preference pairs contain a known incorrect numerical result, opposite entailment answer, or incorrect function result. This narrow source does not establish general human preference quality. The optional `gsm_train` source supplies official training answers; overlap candidates against the frozen suites are excluded. Exact grouped splits are shared between SFT and DPO. Near-match screening does not prove semantic independence.

## Train and select

The Base model's native end token is supervised alongside its answer. User prompts and role prefixes have label `-100`. Sequences are admitted in full, including the end token. The overfit check reloads the saved adapter and requires at least 14/16 entire answers to match, including successful termination. Training on these items is a pipeline diagnostic.

```sh
python -m nlp_course.training sft --config configs/repair-sft.json
```

Full training requires a passing overfit report and a development suite with math, logic and code tasks. Every 50 steps and at the end, it saves `last`, runs the development suite and retains `best` by the equal-weight mean of the three objective accuracies. Ties retain the earlier checkpoint. The Docker engine and the existing `python:3.11-slim` image must be available. Generated code runs only through the existing isolated scorer. `progress.json` preserves the configuration and loss trace before development evaluation.

For DPO, use the same training configuration with `sft_adapter` pointing to the selected SFT directory, `sft_improved_on_dev` set only after comparing against the old SFT on the same development tasks, a fresh output directory, learning rate `5e-6`, and beta `0.1`. Run `python -m nlp_course.training dpo --config <file>`. The preparation must have nonempty verified preference train and validation splits. Reference response log probabilities are computed from the frozen SFT before updates and saved with hashes; only one traversal of complete gradient-accumulation batches is trained. A short final incomplete batch is excluded. Development task scores select the DPO checkpoint. Admit it to the final experiment only if its objective mean improves and no objective category loses more than one correct answer relative to SFT.

## Evaluate locked candidates

Edit the experiment's `models` and `comparisons` after development selection. The old SFT and DPO checkpoints can be added as model entries with an `adapter` path. The official Instruct reference uses its native template. All models get the same tasks, greedy decoding with repetition penalty explicitly fixed at 1.0, and default 256-token budget. Final benchmark outputs must be collected after candidate selection.

```sh
python -m nlp_course.capability generate --experiment-config configs/repair-experiment.json
python -m nlp_course.capability score --experiment-config configs/repair-experiment.json --execute-code
python -m nlp_course.capability render --experiment-config configs/repair-experiment.json
python -m nlp_course.capability review --experiment-config configs/repair-experiment.json
python -m nlp_course.capability import-review --experiment-config configs/repair-experiment.json --ratings <completed.csv>
```

`--model-name` limits generation to one configured model. Resume rejects changed checkpoint, tokenizer, template, suite or generation configuration. An interrupted final JSONL append can be repaired; interior corruption raises an error. Scoring requires exactly one response per model/task. Open-task quality remains pending until human ratings are supplied. `analysis.json` gives strict scores, relaxed numerical matches, truncation, error counts and same-task gains/losses. `capabilities.png` and `errors.png` visualize the objective results and response statuses.

Notebook API credentials are read from the local `NLP_COURSE_API_KEY` environment variable. These local training commands require no external generation API. The owner must revoke the previously exposed key through the provider.
