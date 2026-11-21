"""Freeze version-two banks before GPU training."""

import itertools
import random
from pathlib import Path

from .capability import read, sha
from .capability_data import digest
from .progression import take
from .progression_data import WORDS, check_splits, question_key
from .smollm2_data import freeze

SVAMP_SHA = "5be77703a6d891ae476d7c082787ad361392aa02453b132516cdd5f4e7934e3e"


def compound(row, pool, i):
    rng = random.Random(row["id"])
    records = list(zip(row["words"], rng.sample(range(10, 99), 3)))
    ranked = sorted(records, key=lambda item: item[1])
    listed = ", ".join(f"{word}={value}" for word, value in records)
    cutoff = ranked[0][1]
    templates = {
        "sft": f"Inventory: {listed}. Keep entries above {cutoff}. List names by increasing value.",
        "mine": f"Readings: {listed}. Discard the lowest. Name the survivors from low to high.",
        "dev": f"Scores: {listed}. Which names are second and first, in that order?",
        "holdout": f"Event results: {listed}. The largest score wins. Report silver then gold.",
    }
    prompt = templates[pool]
    selected = [word for word, value in ranked[1:]]
    if i % 2:
        prompt += " Return one uppercase name per line."
        answer = "\n".join(w.upper() for w in selected)
    else:
        prompt += " Use one line, lowercase names separated by a single comma."
        answer = ",".join(selected)
    return dict(row, family=4, prompt=prompt, answer=answer)


def prepare_banks(cfg, train, panels):
    root = Path(cfg["root"])
    old = Path(cfg["prior_inputs"])
    manifest = read(old / "manifest.json")
    old_banks = {k: read(old / "bank" / (k + ".json")) for k in ("sft", "mine", "dev")}
    if any(digest(v) != manifest["banks"][k] for k, v in old_banks.items()):
        raise ValueError("Historical banks changed")
    public = [r for p in panels for r in p]
    if list(map(digest, panels)) != manifest["panels"]:
        raise ValueError("Public questions changed")
    history = public + [r for rows in old_banks.values() for r in rows]
    # All historical GSM training sources, including questions outside progression.
    history.extend(train["gsm8k"])
    seen = {question_key(r) for r in history}
    seen_ids = {r["id"] for r in history}
    blocked_groups = {
        r["source_group"] for r in train["hellaswag"] if r["id"] in seen_ids
    }
    # Public HellaSwag uses another split; exclude source videos as well.
    import pyarrow.parquet as pq

    assets = Path(read(cfg["public_config"])["assets"])
    blocked_groups.update(
        pq.read_table(assets / "hellaswag.parquet", columns=["source_id"])
        .column(0)
        .to_pylist()
    )
    rng = random.Random(cfg["seed"])
    bank = {
        k: take(old_banks[k], n)
        for k, n in [("sft", 1024), ("dev", 128), ("mine", 2048)]
    }
    for pool, rows in bank.items():
        for i, row in enumerate(rows):
            if row["task"] == "ifeval" and (i // 4) % 2:
                rows[i] = compound(row, pool, i)
    # Keep every HellaSwag source video in a single pool, including public validation.
    sources = {r["id"]: r for r in train["hellaswag"]}
    reserved = set(
        pq.read_table(assets / "hellaswag.parquet", columns=["source_id"])
        .column(0)
        .to_pylist()
    )
    extra = list(train["hellaswag"])
    rng.shuffle(extra)
    used_prompts = {question_key(r) for r in public}
    for pool, rows in bank.items():
        original = [r for r in rows if r["task"] == "hellaswag"]
        allowed = []
        for r in itertools.chain((sources[r["id"]] for r in original), extra):
            group, key = r["source_group"], question_key(r)
            if group in reserved or key in used_prompts:
                continue
            reserved.add(group)
            used_prompts.add(key)
            allowed.append(r)
            if len(allowed) == len(original):
                break
        if len(allowed) != len(original):
            raise ValueError(f"Insufficient independent HellaSwag sources for {pool}")
        replacements = iter(allowed)
        bank[pool] = [
            next(replacements) if r["task"] == "hellaswag" else r for r in rows
        ]
    blocked_groups.update(reserved)
    seen.update(used_prompts)
    held = []
    svamp = root / "SVAMP.json"
    if sha(svamp) != SVAMP_SHA:
        raise ValueError("SVAMP source differs from the frozen version")
    math_rows = [
        {
            "id": "svamp-" + r["ID"],
            "task": "gsm8k",
            "dataset": "SVAMP",
            "prompt": r["Body"].strip() + " " + r["Question"].strip(),
            "gold": str(r["Answer"]),
        }
        for r in read(svamp)
    ]
    for task, raw in (
        ("gsm8k", math_rows),
        ("hellaswag", train["hellaswag"]),
        ("piqa", train["piqa"]),
    ):
        rows = list(raw)
        rng.shuffle(rows)
        accepted = []
        for r in rows:
            key = question_key(r)
            group = r.get("source_group")
            if key in seen or (group and group in blocked_groups):
                continue
            if group:
                blocked_groups.add(group)
            accepted.append(r)
            seen.add(key)
            if len(accepted) == 512:
                break
        if len(accepted) != 512:
            raise ValueError(f"Only {len(accepted)} independent {task} rows")
        held.extend(accepted)
    prior_words = {tuple(sorted(r["words"])) for r in history if "words" in r}
    triples = [
        v for v in itertools.combinations(sorted(WORDS), 3) if v not in prior_words
    ]
    for i, triple in enumerate(rng.sample(triples, 512)):
        row = {
            "id": f"v2-holdout-if-{i}",
            "task": "ifeval",
            "words": rng.sample(triple, 3),
        }
        held.append(compound(row, "holdout", i))
    bank["holdout"] = held
    check_splits(bank, panels)
    for k, rows in bank.items():
        freeze(root / "bank" / (k + ".json"), rows)
    freeze(
        root / "manifest.json",
        {
            "config": digest(cfg),
            "banks": {k: digest(v) for k, v in bank.items()},
            "panels": manifest["panels"],
            "svamp": SVAMP_SHA,
            "holdout_math": "SVAMP",
            "public_math": "GSM8K",
        },
    )
