# Reproduce the SmolLM2 public evaluation

The three tasks use the SmolLM2 task definitions at revision d3bb683 and LightEval at ea46419. Inference runs locally with Transformers, BF16 and one RTX 3060 Laptop GPU. A full run needs both public Qwen snapshots and the four project adapters selected in the paired-DPO experiment. The configuration lists the exact revisions and local paths. The runner verifies the model files against the hashes published with the earlier reports. [Configuration](../../configs/smollm2-public.json) · [Prior training](../training.md) · [SmolLM2 reference implementation](https://github.com/huggingface/smollm/tree/d3bb683bc123a1954b289b70dd8119b51c358018/text/evaluation/smollm2)

## Set up

From the repository root, create a separate Python 3.11 environment so the historical evaluation dependencies can coexist with the training environment. The example targets Windows PowerShell and CUDA 12.1.

~~~powershell
uv venv --python 3.11 outputs/smollm2-runtime
$benchPython = "outputs/smollm2-runtime/Scripts/python.exe"
uv pip install --python $benchPython torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python $benchPython -r requirements-smollm2.txt
uv pip install --python $benchPython --no-deps -e .
$env:PYTHONUTF8 = "1"
& $benchPython -m nlp_course.smollm2_eval prepare
~~~

If the two official Qwen checkpoints are not already in the local Hugging Face cache, fetch the revisions from the configuration before entering offline mode. The two official-model runs can be reproduced independently of the four local adapters.

~~~powershell
& $benchPython -c "import json; from huggingface_hub import snapshot_download; cfg=json.load(open('configs/smollm2-public.json')); [snapshot_download(m['repo'],revision=m['revision'],allow_patterns=['*.json','*.txt','*.safetensors','*.model','*.jinja']) for m in cfg['models'].values()]"
~~~

Preparation checks the SHA-256 of the two upstream Python task files and each data asset, loads the original task formatters, verifies labels and freezes the ordered questions. IFEval uses the full public split named train. Its historical registry also initializes a BLEURT dependency at import. Preparation caches and verifies that dependency and the sentence tokenizer for offline scoring. A reachable Hugging Face endpoint can be selected with HF_ENDPOINT before preparation. Downloads and checkpoints stay under ignored directories or the local model cache.

## Run

Run the eight-item checks first. For the formal run, select one task at a time. The published run used batches of eight prompts after the batch-one checks. The choice scorer packs continuation sequences into sub-batches capped at 1,024 padded tokens per forward pass. Every model receives the same split within a task. The instruction model and project models use their respective chat templates. The base model receives raw prompts.

~~~powershell
$env:HF_HUB_OFFLINE = "1"
& $benchPython -m nlp_course.smollm2_eval check --batch 1
& $benchPython -m nlp_course.smollm2_eval run --task piqa --model base --batch 8
& $benchPython -m nlp_course.smollm2_eval run --task piqa --model instruct --batch 8
~~~

Compare those two complete runs with the published PIQA scores of 69.9% and 67.2%. An absolute difference greater than one percentage point triggers inspection of the task snapshot, candidate normalization, chat template, tokenizer and backend. Then finish the matrix. Running each model in a separate process releases the GPU allocation before loading the next model.

~~~powershell
foreach ($task in "piqa","hellaswag","ifeval") {
  foreach ($arm in "base","instruct","frozen","previous_dpo","sft-42","dpo-42") {
    if ($arm -eq "base" -and $task -eq "ifeval") { continue }
    & $benchPython -m nlp_course.smollm2_eval run --task $task --model $arm --batch 8
    if ($LASTEXITCODE -ne 0) { throw "Failed run for $task / $arm" }
  }
}
& $benchPython -m nlp_course.smollm2_eval report
& $benchPython -m nlp_course.smollm2_report verify --rescore-ifeval
~~~

Each completed batch writes its predictions and cost to SQLite in a single transaction. Repeating a command resumes at missing IDs. A changed frozen fingerprint needs a fresh output root. The report requires all 17 result sets, plus the eight-item checks. Prediction files contain question IDs, candidate scores or generated answers, and grading decisions. For IFEval, verification reruns the upstream rule checks on saved responses. Neither operation requires another generation run.

To redraw the figures from published metrics in an environment with the project and its plotting dependencies installed:

~~~sh
python -m nlp_course.smollm2_report render --output outputs/smollm2-figures
~~~

## Protocol details

- PIQA uses validation rows and labels from the archive named by the locked dataset script. HellaSwag uses the author's validation Parquet at revision 218ec52. The GitHub JSONL URL in the historical 6002345 loader is unavailable. Both revisions, the input file hash and the frozen formatted question hash are recorded. Historical JSONL byte identity could not be checked.
- Both choice tasks use zero demonstrations, the pinned LightEval context/continuation tokenizer and the upstream character-normalized likelihood metric. A single leading space is excluded from candidate length. The highest score selects a candidate, with the earliest index resolving ties.
- IFEval uses greedy decoding, an upper bound of 1,280 new tokens and a 2,048-token context window, and stops at the tokenizer's EOS. The inference backend retains each checkpoint's generation settings that apply to greedy decoding. Each strict prompt pass requires every instruction to pass. Instruction-level rates pool checks across prompts. Loose checks use the upstream text transformations.
- The primary comparison is upgraded DPO against continued SFT on the same question IDs. Each 95% percentile interval uses 10,000 paired bootstrap draws and seed 42. For an instruction-level difference, each draw samples complete prompts and recomputes the pooled fraction. Intervals quantify prompt sampling variation conditional on the selected weights. GPU times include synchronized inference and scoring per batch and exclude checkpoint loading, setup and separate smoke checks.
