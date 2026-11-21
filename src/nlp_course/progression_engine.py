"""Plain-text Qwen training and greedy evaluation on one CUDA device."""

import gc
import math
import os
import random

import torch
from .capability import read, sha
from .progression_data import TASKS, final_value, instruction_ok, qa
from .evaluation import parse_number
from .progression_stats import scores
from .smollm2_data import freeze
from .training import preference_loss


def load_model(cfg, adapter=None, trainable=False):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoTokenizer, AutoModelForCausalLM

    base = os.environ.get("PROGRESSION_BASE", cfg["model"])
    args = dict(revision=cfg["revision"], local_files_only=True)
    tok = AutoTokenizer.from_pretrained(base, **args)
    tok.completion_v2 = cfg.get("protocol") == "completion-v2"
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if cfg.get("study") == "hard-pairs-v3":
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"}, **args)
    else:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, **args).cuda()
    if adapter:
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    elif trainable:
        model = get_peft_model(
            model,
            LoraConfig(
                r=cfg["rank"],
                lora_alpha=cfg["alpha"],
                lora_dropout=0,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                task_type="CAUSAL_LM",
            ),
        )
    return model.eval(), tok


def split_tokens(tok, prompt, answer):
    # Match LightEval joint context/continuation tokenization incl. boundary spaces.
    context = prompt.rstrip(" ")
    full = tok.encode(prompt + answer, add_special_tokens=False)
    prefix = tok.encode(context, add_special_tokens=False)
    if len(prefix) == len(full):
        prefix = full[:-1]
    if not prefix or not full[len(prefix) :]:
        raise ValueError("Empty context or continuation")
    return prefix, full[len(prefix) :]


def prompt_for(row):
    return row["prompt"] if "choices" in row else qa(row["prompt"])


def completion_tokens(tok, prompt, answer, task, eos=True):
    if getattr(tok, "completion_v2", False) and task in ("ifeval", "gsm8k"):
        prefix, suffix = [
            tok.encode(text, add_special_tokens=False) for text in (prompt, answer)
        ]
        if not prefix or not suffix or tok.eos_token_id in suffix:
            raise ValueError("Empty prompt/answer or embedded EOS")
    else:
        prefix, suffix = split_tokens(tok, prompt, answer)
    if eos:
        suffix = suffix + [tok.eos_token_id]
    if len(prefix) + len(suffix) > 2048:
        raise ValueError("Training or scoring example exceeds 2048 tokens")
    return prefix, suffix


def logps(model, tok, pairs, eos=True):
    examples = [
        completion_tokens(tok, p, a, task[0] if task else None, eos)
        for p, a, *task in pairs
    ]
    lengths = [len(p) + len(a) for p, a in examples]
    width = max(lengths)
    if width > 2048:
        raise ValueError("Training or scoring example exceeds 2048 tokens")
    ids = torch.full(
        (len(examples), width), tok.pad_token_id, device="cuda", dtype=torch.long
    )
    active = torch.zeros_like(ids, dtype=torch.bool)
    valid = torch.zeros_like(active)
    for i, (p, a) in enumerate(examples):
        ids[i, : lengths[i]] = torch.tensor(p + a, device="cuda")
        active[i, len(p) : lengths[i]] = True
        valid[i, : lengths[i]] = True
    logits = model(input_ids=ids, attention_mask=valid, use_cache=False).logits[:, :-1]
    # Only materialize FP32 vocabulary scores for supervised positions.
    mask = active[:, 1:]
    labels = ids[:, 1:][mask]
    values = -torch.nn.functional.cross_entropy(
        logits[mask].float(), labels, reduction="none"
    )
    sums = values.new_zeros(len(examples)).index_add(0, mask.nonzero()[:, 0], values)
    return sums, mask.sum(1)


@torch.no_grad()
def choices(model, tok, rows):
    pairs = [(r["prompt"], a) for r in rows for a in r["choices"]]
    flat = []
    for start in range(0, len(pairs), 8):
        values, _ = logps(model, tok, pairs[start : start + 8], False)
        flat.extend(values.tolist())
    result = []
    pos = 0
    for r in rows:
        values = flat[pos : pos + len(r["choices"])]
        norm = [
            v / len(a.removeprefix(" "))
            for v, a in zip(values, r["choices"], strict=True)
        ]
        pred = max(range(len(norm)), key=norm.__getitem__)
        result.append(
            dict(pred=pred, normalized=norm, logprob=values, correct=pred == r["gold"])
        )
        pos += len(values)
    return result


@torch.no_grad()
def generate(model, tok, rows, budget, samples=1, sampling=False, temperature=0.85):
    inputs = tok([prompt_for(r) for r in rows], padding=True, return_tensors="pt").to(
        "cuda"
    )
    budget = min(budget, 2048 - inputs.input_ids.shape[1])
    if budget < 1:
        raise ValueError("Prompt too long")
    options = dict(temperature=temperature, top_p=0.95) if sampling else {}
    out = model.generate(
        **inputs,
        do_sample=sampling,
        num_return_sequences=samples,
        max_new_tokens=budget,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        use_cache=True,
        **options,
    )
    result = []
    for ids in out[:, inputs.input_ids.shape[1] :].tolist():
        stop = ids.index(tok.eos_token_id) if tok.eos_token_id in ids else len(ids)
        result.append(
            dict(
                response=tok.decode(ids[:stop], skip_special_tokens=True),
                truncated=stop == len(ids),
            )
        )
    return result


def grade(row, answer):
    if row["task"] == "gsm8k":
        return final_value(answer) == parse_number(row["gold"])
    if "family" in row:
        return instruction_ok(row, answer)
    # Use the locked LightEval registry and its strict algorithm. Per-instruction
    # preparation follows upstream, including dropping false-valued kwargs.
    from lighteval.tasks.extended.ifeval.instructions_registry import INSTRUCTION_DICT

    for name, params in zip(row["instructions"], row["kwargs"], strict=True):
        check = INSTRUCTION_DICT[name](name)
        check.build_description(**{k: v for k, v in params.items() if v})
        if "prompt" in (check.get_instruction_args() or {}):
            check.build_description(prompt=row["prompt"])
        if not answer.strip() or not check.check_following(answer):
            return False
    return True


def evaluate(cfg, model, tok, rows, folder):
    model.eval()
    result = []
    for task in TASKS:
        group = [r for r in rows if r["task"] == task]
        for i in range(0, len(group), cfg["eval_batch"]):
            chunk = group[i : i + cfg["eval_batch"]]
            file = folder / task / f"{i}.json"
            if file.exists():
                batch = read(file)
                if [r["id"] for r in batch] != [r["id"] for r in chunk]:
                    raise ValueError("Cached prediction IDs changed")
            else:
                if task in ("piqa", "hellaswag"):
                    batch = choices(model, tok, chunk)
                else:
                    budget = (
                        cfg["gsm_budget"]
                        if task == "gsm8k"
                        else cfg["instruction_budget"]
                    )
                    batch = generate(model, tok, chunk, budget)
                    for item, r in zip(batch, chunk, strict=True):
                        item["correct"] = bool(grade(r, item["response"]))
                        if task == "gsm8k":
                            item["format_error"] = final_value(item["response"]) is None
                for item, r in zip(batch, chunk, strict=True):
                    item.update(id=r["id"], task=task)
                freeze(file, batch)
            result.extend(batch)
    freeze(folder / "scores.json", scores(result))
    return result


def candidate_pair(row, generated):
    valid = [g for g in generated if not g["truncated"] and g["response"].strip()]
    good = [g["response"] for g in valid if grade(row, g["response"])]
    bad = [g["response"] for g in valid if not grade(row, g["response"])]
    if row["task"] == "gsm8k":
        bad = [s for s in bad if final_value(s) is not None]
    elif bad:
        content = [s for s in bad if all(w.lower() in s.lower() for w in row["words"])]
        bad = content or bad
    if not bad:
        return None
    a, b = min(
        ((a, b) for a in good or [row["answer"]] for b in bad),
        key=lambda p: abs(len(p[0]) - len(p[1])),
    )
    return dict(
        chosen=a,
        rejected=b,
        source="model" if good else "reference",
        greedy_error=not grade(row, generated[0]["response"]),
        model_correct=bool(good),
    )


def mine(cfg, model, tok, bank, folder, hard):
    model.eval()
    pairs = []
    for task in TASKS:
        rows = [r for r in bank if r["task"] == task][: cfg["mining_limit"]]
        accepted = []
        for i in range(0, len(rows), 8):
            chunk = rows[i : i + 8]
            file = folder / task / f"{i}.json"
            if file.exists():
                audit = read(file)
            else:
                torch.manual_seed(cfg["seed"] + i)
                audit = []
                if "choices" in chunk[0]:
                    for r, p in zip(chunk, choices(model, tok, chunk), strict=True):
                        order = sorted(
                            range(len(r["choices"])),
                            key=p["normalized"].__getitem__,
                            reverse=True,
                        )
                        wrong = next(j for j in order if j != r["gold"])
                        pair = dict(
                            chosen=r["answer"],
                            rejected=r["choices"][wrong],
                            source="gold",
                            greedy_error=not p["correct"],
                            model_correct=True,
                            gold_rank=order.index(r["gold"]) + 1,
                        )
                        audit.append(dict(id=r["id"], candidates=p, pair=pair))
                else:
                    budget = cfg["gsm_budget"] if task == "gsm8k" else 96
                    greedy = generate(model, tok, chunk, budget)
                    sampled = generate(
                        model,
                        tok,
                        chunk,
                        budget,
                        3,
                        True,
                        cfg.get("mining_temperature", 0.85),
                    )
                    for j, r in enumerate(chunk):
                        candidates = [greedy[j], *sampled[j * 3 : j * 3 + 3]]
                        audit.append(
                            dict(
                                id=r["id"],
                                candidates=candidates,
                                pair=candidate_pair(r, candidates),
                            )
                        )
                freeze(file, audit)
            for r, entry in zip(chunk, audit, strict=True):
                p = entry["pair"]
                if entry["id"] != r["id"]:
                    raise ValueError("Mining cache mismatch")
                if p and p["chosen"] != p["rejected"]:
                    accepted.append(
                        dict(id=r["id"], task=task, prompt=prompt_for(r), **p)
                    )
            enough = (
                sum(p["greedy_error"] and p["model_correct"] for p in accepted)
                if hard
                else len(accepted)
            )
            if enough >= cfg["pair_target"]:
                break
        if hard:
            accepted.sort(key=lambda p: not (p["greedy_error"] and p["model_correct"]))
        pairs.extend(accepted[: cfg["pair_target"]])
    freeze(folder / "pairs.json", pairs)
    return pairs


def train(cfg, stage, parent, rows, dev, folder, lr, baseline=None):
    rows = list(rows)
    random.Random(cfg["seed"]).shuffle(rows)
    torch.manual_seed(cfg["seed"])
    model, tok = load_model(cfg, parent, True)
    batch = cfg["sft_batch" if stage == "sft" else "dpo_batch"]
    steps = math.ceil(len(rows) / batch)
    checkpoints = {math.ceil(steps * f / 4) for f in (1, 2, 3, 4)}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    folder.mkdir(parents=True, exist_ok=True)
    micro_batch = cfg.get("micro_batch", 4)
    refs = []
    if stage in ("dpo", "hybrid"):
        path = folder / "reference.json"
        identity = sha(parent / "adapter_model.safetensors")
        from .capability_data import digest

        key = (
            digest([identity, rows])
            if not tok.completion_v2
            else digest(
                [identity, rows, cfg["protocol"], tok.backend_tokenizer.to_str()]
            )
        )
        if path.exists():
            saved = read(path)
            if saved["key"] != key:
                raise ValueError("Reference cache identity mismatch")
            refs = saved["values"]
        else:
            with torch.no_grad():
                for i in range(0, len(rows), micro_batch):
                    chunk = rows[i : i + micro_batch]
                    values, _ = logps(
                        model,
                        tok,
                        [
                            (r["prompt"], r[k], r["task"])
                            for r in chunk
                            for k in ("chosen", "rejected")
                        ],
                    )
                    refs.extend(values.reshape(-1, 2).tolist())
            freeze(path, dict(key=key, values=refs))
    resume = folder / "resume.pt"
    start = 0
    if resume.exists():
        from peft import set_peft_model_state_dict

        saved = torch.load(resume, map_location="cpu", weights_only=True)
        set_peft_model_state_dict(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["step"]
    history = []
    for step in range(1, steps + 1):
        if step > start:
            model.train()
            chunk = rows[(step - 1) * batch : step * batch]
            optimizer.zero_grad()
            for j in range(0, len(chunk), micro_batch):
                micro = chunk[j : j + micro_batch]
                if stage in ("sft", "chosen"):
                    sums, n = logps(
                        model,
                        tok,
                        [
                            (prompt_for(r), r["answer"], r["task"])
                            if stage == "sft"
                            else (r["prompt"], r["chosen"], r["task"])
                            for r in micro
                        ],
                    )
                    loss = -(sums / n).mean()
                else:
                    vals, counts = logps(
                        model,
                        tok,
                        [
                            (r["prompt"], r[k], r["task"])
                            for r in micro
                            for k in ("chosen", "rejected")
                        ],
                    )
                    ref = torch.tensor(
                        refs[
                            (step - 1) * batch + j : (step - 1) * batch + j + len(micro)
                        ],
                        device="cuda",
                    )
                    a, b = vals.reshape(-1, 2).T
                    loss = preference_loss(a, b, ref[:, 0], ref[:, 1], cfg["beta"])
                    if stage == "hybrid":
                        loss = loss - 0.1 * (a / counts[::2]).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                (loss * len(micro) / len(chunk)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            from peft import get_peft_model_state_dict

            tmp = folder / "resume.tmp"
            torch.save(
                dict(
                    step=step,
                    adapter=get_peft_model_state_dict(model),
                    optimizer=optimizer.state_dict(),
                ),
                tmp,
            )
            # A checkpoint must be fully scored before moving past its update.
            if step not in checkpoints:
                tmp.replace(resume)
        if step in checkpoints:
            point = folder / f"step-{step}"
            if step >= start and not (point / "summary.json").exists():
                model.save_pretrained(point)
                metrics = scores(evaluate(cfg, model, tok, dev, point / "dev"))
                if stage in ("dpo", "hybrid"):
                    deltas = []
                    with torch.no_grad():
                        for i in range(0, len(rows), micro_batch):
                            vals, _ = logps(
                                model,
                                tok,
                                [
                                    (r["prompt"], r[k], r["task"])
                                    for r in rows[i : i + micro_batch]
                                    for k in ("chosen", "rejected")
                                ],
                            )
                            deltas.extend(
                                (
                                    vals.reshape(-1, 2).cpu()
                                    - torch.tensor(refs[i : i + micro_batch])
                                ).tolist()
                            )
                    metrics["chosen_delta"] = sum(a for a, b in deltas) / len(deltas)
                    metrics["rejected_delta"] = sum(b for a, b in deltas) / len(deltas)
                freeze(point / "summary.json", metrics)
            metrics = read(point / "summary.json")
            eligible = (
                baseline is not None
                and all(metrics[t] >= baseline[t] - 2 - 1e-10 for t in TASKS)
                and metrics["mean"]
                >= baseline["mean"] + (2 if stage == "sft" else 1) - 1e-10
            )
            history.append((eligible, metrics["mean"], -step, point))
            if step > start:
                tmp.replace(resume)
    best = max(history)[-1]
    freeze(
        folder / "selected.json",
        dict(step=best.name, weights=sha(best / "adapter_model.safetensors")),
    )
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return best
