# Can DPO solve both questions when one condition changes?

Both answers in a pair must be correct. Continued SFT solved four of 60 held-out pairs. Upgraded DPO, the frozen SFT B and the previous DPO solved none. Upgraded DPO answered yes to all 40 judgments and got 20 individual answers right. The target was six additional correct pairs against each comparison model.

| Model | Both correct / 60 | Individual correct / 120 | Valid format / 120 | Judgment / 20 pairs | Ordering / 20 pairs | Counting / 20 pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen SFT B | 0 (0%) | 20 (16.7%) | 49 (40.8%) | 0 | 0 | 0 |
| Previous DPO (28 updates) | 0 (0%) | 20 (16.7%) | 55 (45.8%) | 0 | 0 | 0 |
| Continued SFT (update 11) | 4 (6.7%) | 24 (20.0%) | 66 (55.0%) | 4 | 0 | 0 |
| Upgraded DPO (update 6) | 0 (0%) | 20 (16.7%) | 53 (44.2%) | 0 | 0 | 0 |

| DPO comparator | Incorrect to correct | Correct to incorrect | Net pairs | Paired accuracy difference, 95% interval |
| --- | ---: | ---: | ---: | --- |
| Frozen SFT B | 0 | 0 | 0 pairs | 0.0 pp [0.0, 0.0] |
| Continued SFT | 0 | 4 | −4 pairs | −6.7 pp [−13.3, −1.7] |

## Mining preferences

For each prompt, frozen SFT B generated one greedy answer and three sampled answers at temperature 1.25 and top-p 0.98. Eligible negatives were actual, normally terminated, parseable wrong answers. Pairs minimized the difference in answer character count. A solver provided a positive when the model generated no correct answer. The retained judgment prompts had 48 yes and 48 no labels.

| Category | Prompts tried | Eligible error | No eligible error | Retained / target | Yield | Model / solver positives |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Judgment | 168 | 133 | 35 | 96 / 96 | 57.1% | 78 / 18 |
| Ordering | 200 | 66 | 134 | 64 / 64 | 32.0% | 5 / 59 |
| Counting | 256 | 6 | 250 | 6 / 64 | 2.3% | 0 / 6 |

The model generated 2,496 answers for 624 prompts. Of 205 prompts with eligible errors, 166 entered training and 39 were excluded by quotas or label balancing. Counting exhausted four times its target quota and supplied only six pairs to each arm. All four models failed the counting output format on the holdout. Strict scoring cannot distinguish a correct intermediate calculation from an incorrect one in those responses. [Pairs](pairs.json) · [Mining audit](mining.json.gz)

## Training and checkpoint selection

Both arms started from the same adapter and saw the same prompts in the same order for one pass, with seed 42. LoRA rank was 16, alpha 32, dropout zero on Q/K/V/output projections. AdamW used 5e-6 and effective batches of eight, with six pairs in the last batch. DPO used beta 0.1 and fixed reference scores computed before training. Runs used BF16 on an RTX 3060 with 6 GB.

| Update | Continued SFT correct pairs / 30 | DPO correct pairs / 30 | DPO correct response change in log p | DPO wrong response change in log p | DPO preference margin |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 0 | 0 | +0.169 | +0.147 | +0.0022 |
| 11 | 4 | 0 | +0.299 | +0.271 | +0.0029 |
| 16 | 4 | 0 | +0.451 | +0.389 | +0.0063 |
| 21 | 3 | 0 | +0.592 | +0.542 | +0.0050 |

| Training arm | Optimizer updates | Completion and end tokens in loss | Synchronized GPU time |
| --- | ---: | ---: | ---: |
| Continued SFT | 21 | 476 (correct) | 31.1 s |
| Upgraded DPO | 21 | 1,163 (correct and incorrect) | 55.5 s |

Continued SFT selected update 11. DPO selected update 6 under the earliest-tie rule. Its preference margin grew while greedy development accuracy remained at zero pairs. The development criteria did not trigger additional seeds or a hybrid objective. [Checkpoint data and hashes](metrics.json)

## Paired examples

- For holdout-judgment-3, the chain was birch → ash → oak → cedar. Changing the starting fact from elm to birch changed the answer from no to yes. Frozen SFT B and upgraded DPO returned yes/yes. Continued SFT returned no/yes.
- For holdout-judgment-7, the chain was ash → pine → cedar → oak and the starting facts were elm and ash. Continued SFT gave no/no and upgraded DPO gave yes/yes. Each solved one question and neither solved the pair. Upgraded DPO had no complete-pair gain over continued SFT.

## Mathematics and code regression

| Model | Math / 30 | Math gains / losses | Code / 25 | Code gains / losses |
| --- | ---: | ---: | ---: | ---: |
| Frozen SFT B | 9 | — | 6 | — |
| Previous DPO | 10 | 2 / 1 | 6 | 1 / 1 |
| Continued SFT | 9 | 1 / 1 | 5 | 0 / 1 |
| Upgraded DPO | 10 | 1 / 0 | 5 | 0 / 1 |

All arms saw the same 30 GSM8K questions and 25 HumanEval functions under greedy BF16 decoding, batch size four, and a 256-new-token limit. Math required the strict final answer and code ran in the offline Docker grader. The earlier report used FP16 and batch size one, so frozen SFT B was regenerated. Both trained arms gained gsm-237 and lost HumanEval-47. Continued SFT also lost gsm-337. These one- or two-item changes provide limited evidence about general capability. [Per-question results](regression.json)

## Frozen rules and reproduction

Train, development and holdout sets used independent template families. Judgment chains had one, two and three hops. Ordering changed from three to four to five items and reversed one relation while checking for a unique solution. Counting moved from removing items of two colors to adding items of two colors and removing items of three colors. The paired answers had to differ. Both difficulty and templates change across splits.

Scoring trims whitespace, ignores case and removes ordinary spaces for comparison. Judgment accepts yes/no, ordering accepts comma-separated letters and counting accepts integer strings. Extra explanations, contradictory answers and exhausted generation budgets fail. The four arms shared prompts and greedy decoding, a limit of 40 new tokens and Base EOS or ChatML termination. The frozen question SHA-256 is 9431707f5829b4ec76e54f157e7be0eb288f7161e9529094cb821fa7227d1bf3.

Prepare the Qwen snapshot, SFT B, previous 28-update DPO adapter and historical questions in a CUDA environment. The code grader uses the local python:3.11-slim image. [Training assets](../training.md) · [Configuration](../../configs/paired-dpo.json)

```sh
python -m nlp_course.training paired-freeze --config configs/paired-dpo.json
python -m nlp_course.training paired-collect --config configs/paired-dpo.json
python -m nlp_course.training paired-run --config configs/paired-dpo.json
python -m nlp_course.paired_regression --config configs/paired-dpo.json
python -m nlp_course.paired_report --config configs/paired-dpo.json --output docs/paired-dpo-results
```

To redraw the figures from published metrics, install the visualization extra and run the command below. [Checksums](manifest.json)

```sh
python -m nlp_course.paired_report --metrics docs/paired-dpo-results/metrics.json --output outputs/paired-figures
```
