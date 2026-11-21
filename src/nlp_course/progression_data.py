"""Freeze disjoint training sources, development prompts and public acceptance panels."""

import itertools
import json
import os
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from .capability import read, sha
from .capability_data import digest
from .evaluation import parse_number
from .smollm2_data import freeze, upstream_tasks
from .training import group_key

TASKS = ("piqa", "hellaswag", "ifeval", "gsm8k")
WORDS = "acorn amber apple arrow ash atlas badge basil bay bean birch bloom blue boat bowl branch brick bronze brush candle canoe cedar chalk charm cherry clay cloud clover coast cocoa coil coral cork crane cream crest crow cube dawn delta dune dusk eagle earth elm ember fern field finch flame flax flair flint flute foam fog forest fox frost gale gate gem glass globe glow gold grain grape grass green grove".split()


def qa(prompt):
    return "Question: " + prompt + "\nAnswer:"


def final_value(response):
    """Accept marked answers and standalone final lines; reject conflicting claims."""
    claims = []
    pattern = r"(?:####|the (?:final )?answer is\s*:?|final answer\s*:)\s*([^\n]+)"
    for match in re.finditer(pattern, response, re.I):
        value = parse_number(match[1].strip())
        if value is None:
            return None
        claims.append(value)
    lines = response.strip().splitlines()
    if lines:
        value = parse_number(lines[-1])
        if value is not None:
            claims.append(value)
    return claims[0] if claims and len(set(claims)) == 1 else None


def instruction(pool, index, words):
    a, b, c = words
    family = index % 4
    templates = {
        "sft": [
            "List {a}, {b} and {c} in that order, uppercase, one per line.",
            "Print these labels backwards in order: {a}, {b}, {c}. Join them with commas and no spaces.",
            "Return a JSON array containing the strings {a}, {b} and {c} in the listed order.",
            "Make a three-item bulleted list of {a}, {b}, {c}, using '- ' before each word.",
        ],
        "mine": [
            "Convert the labels {a}, {b}, {c} to capitals. Return exactly three lines.",
            "Reverse this sequence: {a} {b} {c}. Separate the output words with commas only.",
            "Encode {a}, {b}, {c} as an ordered JSON list of strings.",
            "Place {a}, {b}, {c} on separate lines as Markdown dash bullets, in the given order.",
        ],
        "dev": [
            "Prepare an uppercase inventory, one label on each line. Labels in shelf order: {a} / {b} / {c}.",
            "The arrivals were {a} then {b} then {c}. Output newest first, comma-separated without spaces.",
            "An API expects a JSON array. Supply these three field names in order: {a}, {b}, {c}.",
            "Create a Markdown checklist of names without checkboxes: {a}, {b}, {c}. Prefix each line with a dash and a space.",
        ],
    }
    answer = [
        "\n".join(w.upper() for w in words),
        ",".join(reversed(words)),
        json.dumps(words),
        "\n".join("- " + w for w in words),
    ][family]
    return dict(
        id=f"if-{pool}-{index}",
        task="ifeval",
        family=family,
        prompt=templates[pool][family].format(a=a, b=b, c=c),
        answer=answer,
        words=words,
    )


def instruction_ok(row, response):
    if row["family"] == 2:
        try:
            return json.loads(response) == list(row["words"])
        except ValueError:
            return False
    return response.strip() == row["answer"]


def question_key(row):
    return re.sub(r"[^\w#]+", " ", group_key(row["prompt"])).strip()


def check_splits(bank, panels):
    seen = {}
    identifiers = set()
    word_sets = set()
    for label, rows in list(bank.items()) + [
        (f"panel-{i}", p) for i, p in enumerate(panels)
    ]:
        for row in rows:
            key = question_key(row)
            if row["id"] in identifiers or seen.setdefault(key, label) != label:
                raise ValueError(f"Overlapping question in {label}: {row['id']}")
            identifiers.add(row["id"])
            if "words" in row:
                words = tuple(sorted(row["words"]))
                if words in word_sets:
                    raise ValueError("Repeated instruction content")
                word_sets.add(words)


def prepare(cfg):
    import pyarrow.parquet as pq

    root = Path(cfg["root"])
    public = read(cfg["public_config"])
    assets = Path(public["assets"])
    os.environ.setdefault("TRANSFORMERS_CACHE", str((assets / "hub").resolve()))
    definitions = upstream_tasks(assets)
    frozen = {
        t: [dict(r, task=t) for r in read(Path(public["root"]) / f"{t}.json")]
        for t in TASKS
        if t != "gsm8k"
    }
    manifests = [
        "docs/capability-results/suite-index.json",
        "docs/repair-results/suite/suite.json",
    ]
    excluded = {
        r["id"]
        for path in manifests
        for r in read(path)["tasks"]
        if r["id"].startswith("gsm-")
    }

    def gsm(path, public_split):
        rows = []
        for i, line in enumerate(Path(path).read_text(encoding="utf8").splitlines()):
            raw = json.loads(line)
            answer = raw.get("answer", raw.get("response"))
            prompt = raw.get("question", raw.get("query"))
            gold = answer.rsplit("####", 1)[-1].strip()
            if parse_number(gold) is None:
                raise ValueError(f"Invalid GSM8K gold at {i}")
            ident = f"gsm-{i}" if public_split else f"train-gsm-{i}"
            if not public_split or ident not in excluded:
                rows.append(
                    dict(
                        id=ident,
                        task="gsm8k",
                        prompt=prompt,
                        gold=gold,
                        **({} if public_split else {"answer": answer}),
                    )
                )
        return rows

    frozen["gsm8k"] = gsm(cfg["gsm_test"], True)
    if len(frozen["gsm8k"]) + len(excluded) != 1319:
        raise ValueError("Wrong GSM8K test split or historical exclusions")
    rng = random.Random(cfg["seed"])
    panels = [[], [], []]

    seen = {question_key(r) for group in frozen.values() for r in group}
    for task, rows in frozen.items():
        groups = defaultdict(list)
        for row in rows:
            groups[question_key(row)].append(row)
        blocks = list(groups.values())
        rng.shuffle(blocks)
        sizes = [0] * 3
        for group in blocks:
            panel = sizes.index(min(sizes))
            panels[panel].extend(group)
            sizes[panel] += len(group)
    for n, panel in enumerate(panels, 1):
        freeze(root / f"panels/{n}.json", panel)
    with zipfile.ZipFile(assets / "piqa.zip") as z:
        lines = z.read("physicaliqa-train-dev/train.jsonl").splitlines()
        labels = z.read("physicaliqa-train-dev/train-labels.lst").splitlines()
        raw_piqa = [
            dict(json.loads(x), label=int(y))
            for x, y in zip(lines, labels, strict=True)
        ]
    if (
        sha(cfg["hellaswag_train"])
        != "cacb12587faa63d7f723a72d61d12bfa94b140446f5a6a0a2e1c6906ab88bf02"
    ):
        raise ValueError("Unexpected HellaSwag training file")
    train = {"gsm8k": gsm(cfg["gsm_train"], False)}
    for task, raw in [
        ("piqa", raw_piqa),
        ("hellaswag", pq.read_table(cfg["hellaswag_train"]).to_pylist()),
    ]:
        train[task] = []
        for i, item in enumerate(raw):
            doc = definitions[task].prompt_function(item, task)
            if not 0 <= doc.gold_index < len(doc.choices):
                raise ValueError("Invalid training label")
            train[task].append(
                dict(
                    **({"source_group": item["source_id"]} if task == "hellaswag" and cfg.get("protocol") == "completion-v2" else {}),
                    id=f"train-{task}-{i}",
                    task=task,
                    prompt=doc.query,
                    choices=doc.choices,
                    gold=doc.gold_index,
                    answer=doc.choices[doc.gold_index],
                )
            )
    if cfg.get("study") == "hard-pairs-v3":
        from .hard_pairs import prepare_splits
        return prepare_splits(cfg, train, panels)
    if cfg.get("protocol") == "completion-v2":
        from .progression_v2_data import prepare_banks
        return prepare_banks(cfg, train, panels)
    bank = {s: [] for s in ("sft", "mine", "dev")}
    counts = {
        "sft": 2 * cfg["train_per_task"],
        "mine": cfg["mining_limit"],
        "dev": 2 * cfg["dev_per_task"],
    }
    for task, rows in train.items():
        rng.shuffle(rows)
        unique = []
        for r in rows:
            key = question_key(r)
            if key not in seen:
                seen.add(key)
                unique.append(r)
        offset = 0
        for pool, n in counts.items():
            batch = unique[offset : offset + n]
            if len(batch) != n:
                raise ValueError(f"Insufficient {task} {pool} rows")
            bank[pool].extend(batch)
            offset += n
    triples = list(itertools.combinations(WORDS, 3))
    rng.shuffle(triples)
    source = iter(triples)
    for pool, n in counts.items():
        bank[pool].extend(
            instruction(pool, i, rng.sample(next(source), 3)) for i in range(n)
        )
    check_splits(bank, panels)
    for name, rows in bank.items():
        freeze(root / f"bank/{name}.json", rows)
    manifest = dict(
        config=digest(cfg),
        excluded=sorted(excluded),
        panels=[digest(p) for p in panels],
        banks={k: digest(v) for k, v in bank.items()},
        sources={k: sha(cfg[k]) for k in ("gsm_test", "gsm_train", "hellaswag_train")},
    )
    freeze(root / "manifest.json", manifest)
    return manifest
