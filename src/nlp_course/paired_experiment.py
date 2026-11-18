"""Local paired preference experiment. Run through nlp_course.training."""

import gc
import math
import random
import time
from pathlib import Path

import torch

from .capability import read, save, sha
from .capability_data import digest
from .paired_tasks import KINDS, PROTOCOL, QUOTAS, assess, build_suite, summarize
from .training import encode, preference_loss, prompt_text, sequence_logps


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(
                f"Experiment changed: choose a fresh output directory ({path})"
            )
    else:
        save(path, value)


def load_model(config, adapter, trainable=False):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise ValueError("A CUDA device is required for this experiment")
    torch.set_num_threads(1)
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(
        config["model"], revision=config["revision"], local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True
    ).cuda()
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=trainable)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, tasks, sampling=False, budget=40):
    prompts = [prompt_text(t["prompt"]) for t in tasks]
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
    stops = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    options = (
        dict(do_sample=True, temperature=1.25, top_p=0.98, num_return_sequences=3)
        if sampling
        else dict(do_sample=False)
    )
    out = model.generate(
        **inputs,
        **options,
        use_cache=True,
        max_new_tokens=budget,
        eos_token_id=stops,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=1.0,
    )
    rows = []
    for tokens in out[:, inputs.input_ids.shape[1] :].tolist():
        end = next((i for i, t in enumerate(tokens) if t in stops), len(tokens))
        rows.append(
            dict(
                response=tokenizer.decode(tokens[:end], skip_special_tokens=True),
                truncated=end == len(tokens),
            )
        )
    return rows


def predict(model, tokenizer, tasks):
    rows = []
    for start in range(0, len(tasks), 8):
        batch = tasks[start : start + 8]
        for task, result in zip(batch, generate(model, tokenizer, batch)):
            rows.append(dict(**task, **result, **assess(task, **result)))
    return rows


def select_pair(task, candidates):
    marked = [dict(**r, **assess(task, **r)) for r in candidates]
    wrong = [
        r["response"].strip() for r in marked if r["format_ok"] and not r["correct"]
    ]
    if not wrong:
        return None
    good = [r["response"].strip() for r in marked if r["correct"]]
    chosen, rejected = min(
        ((g, b) for g in good or [task["answer"]] for b in wrong),
        key=lambda pair: abs(len(pair[0]) - len(pair[1])),
    )
    return dict(
        **task,
        chosen=chosen,
        rejected=rejected,
        chosen_source="model" if good else "solver",
    )


def freeze(config, root):
    suite = build_suite()
    fingerprints = {
        key: sha(Path(config[key]) / "adapter_model.safetensors")
        for key in ["sft", "old_dpo"]
    }
    immutable(
        root / "protocol.json",
        dict(
            version=PROTOCOL,
            config=config,
            weights=fingerprints,
            scoring="full single answer; case-insensitive; malformed, conflicting or length-limited responses fail",
            suite_hash=digest(suite),
            quota=QUOTAS,
            budget=40,
        ),
    )
    immutable(root / "suite.json", suite)
    return suite


def validate_pairs(pairs, train):
    tasks = {task["id"]: task for task in train}
    seen = set()
    for pair in pairs:
        task = tasks.get(pair["id"])
        if task is None or any(pair[key] != value for key, value in task.items()):
            raise ValueError("Pair does not match a frozen training task")
        if pair["id"] in seen:
            raise ValueError("A training prompt contributed more than one pair")
        seen.add(pair["id"])
        positive = assess(task, pair["chosen"], False)
        negative = assess(task, pair["rejected"], False)
        if not positive["correct"] or not negative["format_ok"] or negative["correct"]:
            raise ValueError("Invalid preference labels")
    for kind, quota in QUOTAS.items():
        rows = [pair for pair in pairs if pair["kind"] == kind]
        if len(rows) > quota or len({row["answer"] for row in rows}) < 2:
            raise ValueError(
                f"Insufficient label coverage or exceeded quota for {kind}"
            )


def collect(config, root, suite):
    store = root / "pairs.json"
    if store.exists():
        pairs = read(store)
        validate_pairs(pairs, suite["train"])
        return pairs
    model, tok = load_model(config, config["sft"])
    selected = []
    for kind, quota in QUOTAS.items():
        tasks = [t for t in suite["train"] if t["kind"] == kind]
        collected = []
        for start in range(0, len(tasks), 8):
            batch = tasks[start : start + 8]
            path = root / "mining" / f"{kind}-{start}.json"
            if not path.exists():
                torch.manual_seed(42000 + start)
                greedy = generate(model, tok, batch)
                samples = generate(model, tok, batch, sampling=True)
                records = []
                for j, (t, one) in enumerate(zip(batch, greedy)):
                    answers = [one, *samples[3 * j : 3 * j + 3]]
                    pair = select_pair(t, answers)
                    records.append(
                        dict(
                            id=t["id"],
                            candidates=answers,
                            pair=pair,
                            reason="eligible"
                            if pair
                            else "no unambiguous formatted error",
                        )
                    )
                save(path, records)
            collected.extend(r["pair"] for r in read(path) if r["pair"])
            if kind == "judgment":
                # Preserve both answers even under highly asymmetric error rates.
                if all(
                    sum(r["answer"] == v for r in collected) >= quota // 2
                    for v in ("yes", "no")
                ):
                    break
            elif len(collected) >= quota:
                break
        if kind == "judgment":
            balanced = min(
                quota // 2,
                *(sum(r["answer"] == v for r in collected) for v in ("yes", "no")),
            )
            collected = [
                r
                for v in ("yes", "no")
                for r in [x for x in collected if x["answer"] == v][:balanced]
            ]
        if not collected:
            raise ValueError(f"No usable coverage for {kind}; inspect mining records")
        selected.extend(collected[:quota])
        print(kind, "selected", len(collected[:quota]), flush=True)
    validate_pairs(selected, suite["train"])
    save(store, selected)
    return selected


def diagnostics(rows):
    """Contrast the correct response with the sibling's now-incorrect answer."""
    return [
        dict(**row, chosen=row["answer"], rejected=rows[i ^ 1]["answer"])
        for i, row in enumerate(rows)
    ]


def encoded_pairs(tok, rows):
    return [
        (
            encode(tok, r["prompt"], r["chosen"], 512),
            encode(tok, r["prompt"], r["rejected"], 512),
        )
        for r in rows
    ]


@torch.inference_mode()
def paired_logps(model, examples, device="cuda"):
    return [
        (
            sequence_logps(model, a, device).item(),
            sequence_logps(model, b, device).item(),
        )
        for a, b in examples
    ]


def train_arm(config, root, mode, seed, rows, dev, base_count):
    folder = root / f"{mode}-{seed}"
    ordered = rows.copy()
    random.Random(seed).shuffle(ordered)
    order = [r["id"] for r in ordered]
    immutable(folder / "order.json", order)
    report_path = folder / "training.json"
    if report_path.exists():
        return read(report_path)
    torch.manual_seed(seed)
    model, tok = load_model(config, config["sft"], True)
    examples = encoded_pairs(tok, ordered)
    monitor = encoded_pairs(tok, diagnostics(dev))
    reference = paired_logps(model, examples) if mode != "sft" else []
    monitor_ref = paired_logps(model, monitor)
    adam = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=5e-6
    )
    steps = math.ceil(len(examples) / 8)
    checkpoints = {math.ceil(steps * q / 4) for q in (1, 2, 3, 4)}
    history = []
    tokens = 0
    best = -1
    triggered = False
    previous_signal = False
    train_seconds = 0.0
    for step in range(1, steps + 1):
        model.train()
        batch = examples[(step - 1) * 8 : step * 8]
        torch.cuda.synchronize()
        start = time.perf_counter()
        adam.zero_grad()
        for j, (a, b) in enumerate(batch):
            na = sum(t != -100 for t in a["labels"])
            lp = sequence_logps(model, a, "cuda")
            tokens += na
            loss = -lp.mean() / na
            if mode != "sft":
                neg = sequence_logps(model, b, "cuda")
                tokens += sum(t != -100 for t in b["labels"])
                pair_loss = preference_loss(
                    lp, neg, *reference[(step - 1) * 8 + j], 0.1
                )
                loss = pair_loss + 0.1 * loss if mode == "hybrid" else pair_loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite objective")
            (loss / len(batch)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        adam.step()
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - start
        if step not in checkpoints:
            continue
        model.eval()
        predictions = predict(model, tok, dev)
        metrics = summarize(predictions)
        save(folder / f"dev-{step}.json", predictions)
        values = paired_logps(model, monitor)
        deltas = [
            sum(v[k] - r[k] for v, r in zip(values, monitor_ref)) / len(values)
            for k in (0, 1)
        ]
        margin = 0.1 * (deltas[0] - deltas[1])
        prev = history[-1] if history else dict(margin=0, chosen_delta=0)
        signal = (
            margin > prev["margin"]
            and deltas[0] < prev["chosen_delta"]
            and metrics["both"] <= base_count
        )
        triggered |= signal and previous_signal
        previous_signal = signal
        history.append(
            dict(
                step=step,
                chosen_delta=deltas[0],
                rejected_delta=deltas[1],
                margin=margin,
                both=metrics["both"],
                single=metrics["single"],
                format_ok=metrics["format_ok"],
            )
        )
        if metrics["both"] > best:
            best = metrics["both"]
            model.save_pretrained(folder / "best")
            best_step = step
        print(mode, seed, step, "dev groups", metrics["both"], flush=True)
    result = dict(
        mode=mode,
        seed=seed,
        best_step=best_step,
        best=best,
        history=history,
        supervised_tokens=tokens,
        train_gpu_wall_seconds=train_seconds,
        order_hash=digest(order),
        hybrid_trigger=triggered,
    )
    save(report_path, result)
    return result


def evaluate_arm(config, root, name, path, tasks, split):
    destination = root / split / f"{name}.json"
    immutable(
        root / "evaluation-inputs" / split / f"{name}.json",
        {"adapter": sha(Path(path) / "adapter_model.safetensors"), "tasks": digest(tasks)},
    )
    if not destination.exists():
        model, tok = load_model(config, path)
        save(destination, predict(model, tok, tasks))
    return summarize(read(destination))


def difference(candidate, baseline, seed=42):
    keys = sorted(candidate)
    if set(keys) != set(baseline):
        raise ValueError("Unmatched groups")
    delta = [int(candidate[k]) - int(baseline[k]) for k in keys]
    rng = random.Random(seed)
    draws = sorted(
        sum(rng.choices(delta, k=len(keys))) / len(keys) for _ in range(10000)
    )
    return dict(
        improved=delta.count(1),
        regressed=delta.count(-1),
        net=sum(delta),
        interval=[draws[249], draws[9749]],
        improved_ids=[k for k, d in zip(keys, delta) if d == 1],
        regressed_ids=[k for k, d in zip(keys, delta) if d == -1],
    )


def run_comparison(config, root, suite, pairs):
    immutable(root / "training-input.json", {"pairs": digest(pairs), "suite": digest(suite)})
    gc.collect()
    torch.cuda.empty_cache()
    arms = dict(frozen=config["sft"], previous_dpo=config["old_dpo"])
    dev = {
        name: evaluate_arm(config, root, name, path, suite["dev"], "dev")
        for name, path in arms.items()
    }
    results = {}
    for seed in (42, 43, 44):
        for mode in ("sft", "dpo", "hybrid"):
            if mode == "hybrid" and (
                seed != 42 or not results["dpo-42"]["hybrid_trigger"]
            ):
                continue
            name = f"{mode}-{seed}"
            results[name] = train_arm(
                config, root, mode, seed, pairs, suite["dev"], dev["frozen"]["both"]
            )
            arms[name] = str(root / name / "best")
        if (
            seed == 42
            and results["dpo-42"]["best"]
            - max(results["sft-42"]["best"], dev["frozen"]["both"])
            < 3
        ):
            break
    immutable(
        root / "selection.json", dict(arms=arms, training=results, dev_baselines=dev)
    )
    final = {
        name: evaluate_arm(config, root, name, path, suite["holdout"], "holdout")
        for name, path in arms.items()
    }
    comparisons = {
        name: {
            control: difference(final[name]["groups"], final[control]["groups"])
            for control in ("frozen", "sft-" + name.split("-")[-1])
        }
        for name in results
        if not name.startswith("sft")
    }
    save(root / "results.json", dict(final=final, comparisons=comparisons))
