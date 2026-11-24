# Does standard DPO improve the SFT checkpoint?

**Holdout details.** The 511 training pairs required a model error, greater likelihood for the wrong completion under training tokenization, and a token-length ratio at most 1.5. The seed was 42. DPO selected update 8 at 0.00002 and the supervised control selected update 32 at 0.00008. Holdout gains over frozen SFT were two PIQA and five SVAMP answers, with zero net change on HellaSwag and the instruction tasks. Among 192 SVAMP answers, unique numeric extraction failed 109 times for SFT, 102 for DPO and 29 for the supervised control. On the public RTX 3060 run, DPO-minus-SFT changes were −0.27, +0.06, +0.74 and +0.08 points for PIQA, HellaSwag, strict IFEval and GSM8K respectively.

This follow-up used Qwen2.5-0.5B Base after one round of supervised fine-tuning (SFT). Each experimental branch started from the same SFT adapter. Seed 42 was used throughout. [Reproduction commands](../hard-pairs-v3.md)

## Training comparison

The frozen SFT model supplied the errors for 511 preference pairs. For each selected pair, the model answered the prompt incorrectly and assigned the wrong answer a higher sequence log probability under the training encoding. The longer answer could contain at most 1.5 times as many tokens as the shorter answer.

Standard DPO learned from both answers. The supervised control learned from the correct answers alone. Both had the same sample order, learning-rate grid, and maximum update budget. Development-set selection chose DPO at step 8 with a learning rate of 0.00002 and the control at step 32 with a learning rate of 0.00008.

## Results

| Four-task mean accuracy | SFT | DPO | Correct-answer-only control | DPO minus SFT, paired 95% CI |
| --- | ---: | ---: | ---: | --- |
| New holdout, 768 prompts | 44.66% | 45.57% | 52.73% | +0.91 percentage points [+0.13, +1.82] |
| Public suite, 13,680 prompts | 43.49% | 43.64% | Not evaluated | +0.15 percentage points [−0.35, +0.66] |

On the holdout, DPO gained two net correct answers on PIQA and five on SVAMP. HellaSwag and the custom instruction task had zero net change. The gain fell below the prespecified one-point target, and the adjusted one-sided lower bound was zero. The supervised control exceeded DPO by 7.16 points (paired 95% CI +5.08 to +9.24).

Answer extraction accounted for a substantial difference between the branches. Among the 192 SVAMP prompts, the scorer could not extract a unique valid number from 109 SFT responses, 102 DPO responses, and 29 control responses. These counts identify a formatting difference. Arithmetic correctness is assessed separately by comparing the extracted number to the reference.

On the public suite, DPO changed accuracy by −0.27 points on PIQA, +0.06 on HellaSwag, +0.74 on strict IFEval, and +0.08 on GSM8K. SFT and DPO were evaluated on the same RTX 3060 after their weights were locked. The intervals describe variation across prompts for this seed and these checkpoints. [Per-task gains, losses, intervals, and weight hashes](metrics.json)
