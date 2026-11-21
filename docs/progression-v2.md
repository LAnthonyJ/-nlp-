# Reproduce the completion-boundary experiment

The final chain uses Qwen2.5-0.5B Base, one SFT adapter and one preference update stage. The chosen-only control and optional hybrid train separately from the selected SFT adapter. All branches use seed 42. Training uses the BF16 environment from [the first progression experiment](progression.md).

## Prepare on CPU

Restore the previous frozen banks at `outputs/progression-v1` and the returned first experiment at `outputs/progression-return/outputs/progression-v1`. Restore the pinned Base tokenizer in the Hugging Face cache or point `PROGRESSION_BASE` to the local snapshot. Use the archived SmolLM2 preparation environment for `progression-prepare`.

```sh
export PYTHONPATH=src
python -m nlp_course.training progression-audit --config configs/progression-v2.json
curl -fL https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json -o outputs/progression-v2/SVAMP.json
python -m nlp_course.training progression-prepare --config configs/progression-v2.json
python -m unittest discover -s tests
```

Preparation verifies the SHA-256 of the SVAMP source and freezes 512 public-source PIQA prompts, 512 HellaSwag prompts, 512 original instruction exercises and 512 SVAMP questions. SVAMP uses the existing final-number grader and greedy decoding budget. Some internal batch files use `gsm8k` as the arithmetic task key; the source and report identify the held-out dataset as SVAMP. The official 13,680-prompt public comparison retains GSM8K. Numeric and punctuation normalization can detect exact and near duplicates but cannot certify absence from Base pretraining. HellaSwag source videos and instruction word triples are disjoint across v2 splits. The custom holdout measures an order-and-filter operation with two output formats. [SVAMP dataset and citation](https://github.com/arkilpatel/SVAMP)

The audit writes decoded supervised spans and historical labels under `outputs/progression-v2/audit`. It verifies the original adapter hash and the shuffled row identity against the archived DPO reference. Keep both the data and manifest when transferring the experiment. The original round-2 DPO trajectory and development predictions, round-1 SFT step-32 weights, original development bank and public panels are required for the ablation and comparisons.

## Train and evaluate on one GPU

Run from the repository root on Linux. Use the pinned Base snapshot, LightEval instruction registry and NLTK assets specified in the first experiment. Train first, then evaluate the locked weights. Both commands can resume from the same directory.

```sh
export PYTHONPATH=src:upstream NLTK_DATA=upstream/nltk_data PROGRESSION_BASE=base
export TOKENIZERS_PARALLELISM=false
python -m nlp_course.training progression-run --config configs/progression-v2.json
python -m nlp_course.training progression-public --config configs/progression-v2.json
```

The run first compares step 64 under repaired and legacy supervision on the same old development prompts. Next it fits SFT on 4,096 examples, mines up to 256 pairs per task, then fits standard DPO and chosen-only arms with matched order and updates. Hybrid training requires a failed development gate and two successive checkpoints where the chosen log probability falls, the preference margin grows and mean generation accuracy stays at or below the SFT baseline. All checkpoint and arm choices precede the holdout evaluation.

## Verify the return and generate figures

Transfer the output directory using a SHA-256-verified archive. With the locked preparation environment, its LightEval registry, NLTK files and Matplotlib, run the command below. It checks model hashes, regrades individual answers and reconstructs the paired intervals before writing the report, cases, predictions, costs and two figures. Compare the returned manifest to the locally frozen copy before replacing any files. Shut down the rented instance after the return is verified.

```sh
python -m nlp_course.training progression-report --config configs/progression-v2.json
```
