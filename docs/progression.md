# Reproduce the Base → SFT → DPO experiment

Each complete chain starts at the pinned Qwen2.5-0.5B Base. DPO updates the selected SFT LoRA once, with its reference log probabilities fixed before the first update. Each chain has one SFT stage and one DPO stage. Development retries may reuse a successful SFT. A new attempt starts its DPO from that SFT. See the [configuration](../configs/progression.json) and the [public task preparation guide](smollm2-results/reproduce.md).

## Freeze the inputs

From the repository root, use the Python 3.11 SmolLM2 preparation environment from the public task guide. Run its prepare command first. Place GSM8K train and test JSONL at the configuration paths. The reader accepts the official question/answer fields and the project's archived query/response representation with the same original order. Download HellaSwag train from the author revision below. PIQA train comes from the same locked archive as validation.

~~~sh
curl -fL https://huggingface.co/datasets/Rowan/hellaswag/resolve/218ec52e09a7e7462a5400043bb9a69a41d06b76/data/train-00000-of-00001.parquet -o outputs/smollm2-upstream/hellaswag-train.parquet
PYTHONPATH=src python -m nlp_course.training progression-prepare --config configs/progression.json
~~~

The input manifest records exact file and ordered-bank hashes. The public partitions contain the original PIQA and HellaSwag validation splits, IFEval prompts, and GSM8K test with the 60 previously measured IDs excluded. Base receives the identical raw continuation or plain-text question/answer prompt as SFT and DPO. This is an English, benchmark-targeted adaptation experiment. Scores from the earlier model-specific templates are historical context.

Training and mining have distinct prompts. Normalization groups case, whitespace, punctuation and numeric substitutions. It cannot detect every semantic paraphrase. The independently authored instruction development templates and word triples are held out. These exercises measure compliance with four structured transformations and have a different distribution from public IFEval.

## Run on the GPU

The cloud run uses Linux, Python 3.12, CUDA 12.8 PyTorch 2.8.0, Transformers 4.51.3, PEFT 0.15.2, Accelerate 1.6.0, NumPy 2.3.2, langdetect 1.0.9 and NLTK 3.9.1. The RTX 5090 needs a Blackwell-compatible PyTorch build. Install the matching CUDA build before installing the project. The frozen inputs can be copied from the preparation machine. Copy the three upstream IFEval instructions modules from the pinned LightEval checkout to upstream/lighteval/tasks/extended/ifeval and its English punkt_tab files to upstream/nltk_data/tokenizers/punkt_tab/english. The CPU verifier additionally uses the original LightEval metric in the preparation environment.

Also copy the verified English punkt/english.pickle into upstream/nltk_data/tokenizers/punkt/english.pickle. The locked instruction registry checks for this legacy directory during import, even with the newer English tables installed.

~~~sh
export PYTHONPATH=src:upstream NLTK_DATA=upstream/nltk_data
export PROGRESSION_BASE=base TOKENIZERS_PARALLELISM=false
huggingface-cli download Qwen/Qwen2.5-0.5B --revision 060db6499f32faf8b98477b0a26969ef7d8b9987 --local-dir base
python -m nlp_course.training progression-run --config configs/progression.json
python -m nlp_course.training progression-public --config configs/progression.json
python -m nlp_course.training progression-report --config configs/progression.json
~~~

The runner checks the base weight hash and BF16 support, then learns 16 training answers in a disposable adapter and reloads it. Formal SFT starts fresh. One training epoch uses shuffled examples, completion-only labels, native EOS, gradient clipping at 1, AdamW and effective batches of 32 for SFT and 16 for DPO. The optimizer sees all examples including a short final batch. The largest development mean at the four checkpoints selects the adapter, with earlier steps breaking ties. Generation uses greedy decoding with a 2,048-token total context. GSM8K has a 384-new-token cap and IFEval has a 1,280-token cap. PIQA and HellaSwag use character-normalized continuation likelihood.

Every completed optimizer update has resumable adapter and optimizer state. Each mining/evaluation batch is immutable. Reissue the same command with the same inputs after infrastructure interruption. Changed data or configuration needs a new output root. Keep the complete output directory to retain candidate responses, preferences, fixed reference scores, checkpoint history, final adapters and acceptance records.

## Read the result

A stage passes the development gate when the equal-weight mean increases by at least 2 percentage points for SFT or 1 for DPO, with no individual decline above 2 points. Only two passing stages open a formal panel. Formal acceptance also requires positive one-sided bootstrap lower bounds, using a tail probability of 0.05/6 for two stages and up to three attempts. Each of 50,000 draws resamples paired question changes within each task and combines task differences at equal weight. Uncertainty is conditional on the trained weights and the sampled prompts.

The full-suite run locks the last completed chain chronologically, after the retry loop. It reports all public questions, including any previously opened panels, as descriptive measurements. It does not replace the registered acceptance decision. Historical project runs had already evaluated the public PIQA, HellaSwag and IFEval prompts. Independently reproduce scores by copying cloud outputs to a second root and running the verifier against the original frozen input root in the SmolLM2 preparation environment.

~~~python
from pathlib import Path
from nlp_course.progression_report import verify_return
verify_return(Path("outputs/progression-return/outputs/progression-v1"), Path("outputs/progression-v1"))
~~~

For verifier setup, set NLTK_DATA to the prepared nltk_data directory, TRANSFORMERS_CACHE to the prepared hub directory, and HF_HUB_OFFLINE to 1. The verifier recomputes all choice decisions and numeric matches, reruns the original IFEval metric, checks all prompt IDs, rebuilds paired intervals and hashes the returned model adapters.
