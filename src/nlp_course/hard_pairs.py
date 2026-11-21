"""Preregister, mine and evaluate standard DPO on reference-model errors."""

import argparse
import itertools
import json
import random
from pathlib import Path

import torch

from .capability import read, sha
from .capability_data import digest
from .progression_data import TASKS, WORDS, check_splits, prepare, question_key
from .progression_engine import grade, load_model, logps, prompt_for
from .progression_stats import compare, scores
from .progression_v2 import history_rows, predict, train_arm, weights
from .progression_v2_data import SVAMP_SHA
from .smollm2_data import freeze


def instruction_row(words, split, index):
    a, b, c = words
    family = index % 5
    rules = {
        "dev": [
            "Use capitals and give one item per line: {a}, {b}, {c}.",
            "The queue was {a}, {b}, {c}. Print the reversed queue using commas without spaces.",
            "Serialize {a}, {b}, {c} as a JSON array of strings in their given order.",
            "Turn {a}, {b}, {c} into three Markdown list items with a dash and a space.",
            "Grades are {records}. Return second place then first place, comma-separated without spaces.",
        ],
        "holdout": [
            "The three sign labels are {a} / {b} / {c}. Transcribe each on its own line in uppercase.",
            "Read the roster from last to first: {a} | {b} | {c}. Use only commas between names.",
            "Output an ordered JSON string array for these fields: {a}; {b}; {c}.",
            "Write a Markdown dash list with one entry per line for {a}, {b} and {c}, in that sequence.",
            "The largest tally wins. Tallies: {records}. Write runner-up,winner in lowercase without spaces.",
        ],
    }
    values = random.Random(f"{split}-{index}").sample(range(10, 99), 3)
    ranked = sorted(zip(values, words))
    formats = [
        "\n".join(w.upper() for w in words),
        ",".join(reversed(words)),
        json.dumps(words),
        "\n".join("- " + w for w in words),
        ranked[1][1] + "," + ranked[2][1],
    ]
    records = ", ".join(f"{w}={n}" for w, n in zip(words, values))
    return {
        "id": f"hard-{split}-if-{index}",
        "task": "ifeval",
        "family": family,
        "words": words,
        "prompt": rules[split][family].format(a=a, b=b, c=c, records=records),
        "answer": formats[family],
    }


def prepare_splits(cfg, train, panels):
    import pyarrow.parquet as pq

    root = Path(cfg["root"])
    v2 = Path(cfg["local_v2"])
    historic = [r for p in panels for r in p]
    for folder in (Path(cfg["prior_inputs"]), v2):
        manifest = read(folder / "manifest.json")
        for name, checksum in manifest["banks"].items():
            rows = read(folder / "bank" / (name + ".json"))
            if digest(rows) != checksum:
                raise ValueError("Historical bank changed")
            historic.extend(rows)
    existing = {question_key(r) for r in historic + train["gsm8k"]}
    ids = {r["id"] for r in historic}
    assets = Path(read(cfg["public_config"])["assets"])
    groups = set(
        pq.read_table(assets / "hellaswag.parquet", columns=["source_id"])
        .column(0)
        .to_pylist()
    )
    groups.update(r["source_group"] for r in train["hellaswag"] if r["id"] in ids)
    if sha(v2 / "SVAMP.json") != SVAMP_SHA:
        raise ValueError("SVAMP source changed")
    svamp = read(v2 / "SVAMP.json")
    body = lambda item: question_key({"prompt": item["Body"]})
    used_stories = {body(x) for x in svamp if "svamp-" + x["ID"] in ids}
    used_words = {tuple(sorted(r["words"])) for r in historic if "words" in r}
    rng = random.Random(903)
    triples = [
        v for v in itertools.combinations(sorted(WORDS), 3) if v not in used_words
    ]
    rng.shuffle(triples)
    for task in ("piqa", "hellaswag"):
        rng.shuffle(train[task])
    rng.shuffle(svamp)
    banks = {}
    for split, n in (("dev", cfg["dev_per_task"]), ("holdout", cfg["final_per_task"])):
        rows = []
        for task in ("piqa", "hellaswag", "gsm8k"):
            selected = []
            for raw in svamp if task == "gsm8k" else train[task]:
                if task == "gsm8k":
                    if body(raw) in used_stories:
                        continue
                    r = {
                        "id": "svamp-" + raw["ID"],
                        "prompt": raw["Body"].strip() + " " + raw["Question"].strip(),
                        "task": task,
                        "gold": str(raw["Answer"]),
                        "dataset": "SVAMP",
                    }
                else:
                    r = raw
                key = question_key(r)
                if r["id"] in ids or key in existing or r.get("source_group") in groups:
                    continue
                existing.add(key)
                ids.add(r["id"])
                if task == "hellaswag":
                    groups.add(r["source_group"])
                if task == "gsm8k":
                    used_stories.add(body(raw))
                selected.append(r)
                if len(selected) == n:
                    break
            if len(selected) != n:
                raise ValueError(
                    f"Insufficient unseen groups for {task} in {split}: {len(selected)}"
                )
            rows.extend(selected)
        rows.extend(
            instruction_row(rng.sample(triples.pop(), 3), split, i) for i in range(n)
        )
        banks[split] = rows
    check_splits(banks, panels)
    for name, data in banks.items():
        freeze(root / "bank" / (name + ".json"), data)
    freeze(
        root / "design.json",
        {
            "config": digest(cfg),
            "banks": {k: digest(v) for k, v in banks.items()},
            "objective": "standard DPO",
            "selection": "four-task mean, phase gain >=1 pp and all tasks >= -2 pp first; then earlier step and lower learning rate",
            "formal_gain_pp": 1,
            "require_positive_adjusted_lower": True,
        },
    )


def collect(cfg, parent):
    root, archive = Path(cfg["root"]), Path(cfg["returned"])
    bank = read(archive / "bank/mine.json")
    if (
        digest(bank) != read(archive / "manifest.json")["banks"]["mine"]
        or weights(parent, cfg) != read(archive / "mining/identity.json")["parent"]
    ):
        raise ValueError("Mining lineage mismatch")
    index = {r["id"]: r for r in bank}
    rng = random.Random(cfg["seed"])
    model, tok = load_model(cfg, parent)
    collected = []
    for task in TASKS:
        eligible = []
        for f in sorted((archive / "mining" / task).glob("*.json")):
            for entry in read(f):
                row = index[entry["id"]]
                if "choices" in row:
                    decision = entry["candidates"]
                    if decision["correct"]:
                        continue
                    pos = row["answer"]
                    neg = row["choices"][decision["pred"]]
                else:
                    responses = entry["candidates"]
                    greedy = responses[0]
                    neg = greedy["response"]
                    if greedy["truncated"] or not neg.strip() or grade(row, neg):
                        continue
                    positives = [
                        x["response"]
                        for x in responses[1:]
                        if not x["truncated"] and grade(row, x["response"])
                    ]
                    if not positives and task == "gsm8k":
                        continue
                    pos = min(
                        positives or [row["answer"]],
                        key=lambda s: abs(len(s) - len(neg)),
                    )
                    if not grade(row, pos):
                        continue
                lengths = [
                    len(tok.encode(x, add_special_tokens=False)) for x in (pos, neg)
                ]
                if min(lengths) == 0 or max(lengths) > 1.5 * min(lengths):
                    continue
                eligible.append(
                    {
                        "id": row["id"],
                        "task": task,
                        "prompt": prompt_for(row),
                        "chosen": pos,
                        "rejected": neg,
                        "family": row.get("family", 0),
                        "source": "gold"
                        if "choices" in row
                        else "model"
                        if positives
                        else "reference",
                    }
                )
        rng.shuffle(eligible)
        buckets = {}
        with torch.inference_mode():
            for r in eligible:
                v, _ = logps(
                    model,
                    tok,
                    [(r["prompt"], r[k], task) for k in ("chosen", "rejected")],
                )
                margin = (v[0] - v[1]).item()
                if margin >= 0:
                    continue
                buckets.setdefault(r["family"], []).append(
                    dict(r, reference_margin=margin)
                )
        for rows in buckets.values():
            rows.sort(
                key=lambda r: (r["source"] == "reference", abs(r["reference_margin"]))
            )
        picked = []
        for i in range(max(map(len, buckets.values()), default=0)):
            for k in sorted(buckets):
                if i < len(buckets[k]):
                    picked.append(buckets[k][i])
        selected = picked[: cfg["pairs_per_task"]]
        if len(selected) < 32:
            raise ValueError(f"Insufficient eligible {task} pairs: {len(selected)}")
        collected.extend(selected)
        print(f"{task}: {len(selected)} of {len(picked)} model errors", flush=True)
    freeze(root / "pairs.json", collected)
    freeze(
        root / "pair-manifest.json",
        {
            "config": digest(cfg),
            "parent": weights(parent, cfg),
            "rows": digest(collected),
        },
    )
    return collected


def run(cfg):
    root = Path(cfg["root"])
    design = read(root / "design.json")
    if design["config"] != digest(cfg):
        raise ValueError("Preregistered config changed")

    def bank(name):
        rows = read(root / "bank" / (name + ".json"))
        if digest(rows) != design["banks"][name]:
            raise ValueError("Frozen bank changed")
        return rows

    archive = Path(cfg["returned"])
    sft = archive / "sft/step-64"
    if weights(sft, cfg) != read(archive / "weights-lock.json")["arms"]["sft"]["sha"]:
        raise ValueError("SFT changed")
    dev = bank("dev")
    pairs = read(root / "pairs.json")
    receipt = read(root / "pair-manifest.json")
    if receipt != {
        "config": digest(cfg),
        "parent": weights(sft, cfg),
        "rows": digest(pairs),
    }:
        raise ValueError("Mined pairs changed")
    sft_dev = predict(cfg, sft, dev, root / "sft-dev")
    baseline = scores(sft_dev)
    arms = {"sft": sft}
    for stage in ("dpo", "chosen"):
        trials = []
        for lr in cfg["learning_rates"]:
            f = root / f"{stage}-{lr:g}"
            local = dict(cfg, dpo_lr=lr)
            ckpt = train_arm(local, stage, sft, pairs, dev, f, baseline)
            result = compare(sft_dev, history_rows(ckpt / "dev"), 1)
            trials.append(
                (
                    result["passed"],
                    result["pp"],
                    -int(ckpt.name.split("-")[1]),
                    -lr,
                    ckpt,
                )
            )
            print(stage, lr, ckpt, result["pp"], flush=True)
        arms[stage] = max(trials)[-1]
    freeze(
        root / "weights-lock.json",
        {
            k: {"path": str(v.as_posix()), "sha": weights(v, cfg)}
            for k, v in arms.items()
        },
    )
    for split, rows in (("holdout", bank("holdout")),):
        preds = {k: predict(cfg, v, rows, root / split / k) for k, v in arms.items()}
        result = {
            "scores": {k: scores(v) for k, v in preds.items()},
            "dpo": compare(preds["sft"], preds["dpo"], 1, True),
            "control": compare(preds["chosen"], preds["dpo"], 0, True),
        }
        freeze(root / split / "results.json", result)
        print(split, result, flush=True)


def public(cfg):
    from langdetect import DetectorFactory

    DetectorFactory.seed = cfg["seed"]
    root = Path(cfg["root"])
    locked = read(root / "weights-lock.json")
    panels = []
    prior = Path(cfg["prior_inputs"])
    manifest = read(prior / "manifest.json")
    for i, expected in enumerate(manifest["panels"], 1):
        panel = read(prior / "panels" / f"{i}.json")
        if digest(panel) != expected:
            raise ValueError("Public panel changed")
        panels.extend(panel)
    predictions = {}
    for k in ("sft", "dpo"):
        adapter = Path(locked[k]["path"])
        if weights(adapter, cfg) != locked[k]["sha"]:
            raise ValueError("Locked adapter changed")
        predictions[k] = predict(cfg, adapter, panels, root / "public" / k)
    freeze(
        root / "public/results.json",
        {
            "scores": {k: scores(v) for k, v in predictions.items()},
            "dpo": compare(predictions["sft"], predictions["dpo"], 1, True),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "collect", "run", "public"])
    args = parser.parse_args()
    cfg = read("configs/hard-pairs-v3.json")
    if args.command == "collect":
        collect(cfg, Path(cfg["returned"]) / "sft/step-64")
    elif args.command == "public":
        public(cfg)
    else:
        (prepare if args.command == "prepare" else run)(cfg)
