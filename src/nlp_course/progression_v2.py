"""Run the versioned completion-boundary ablation and prospective experiment."""

import os
import random
import time
from collections import Counter
from pathlib import Path

import torch

from .capability import read, sha
from .capability_data import digest
from .progression import take
from .progression_data import TASKS
from .progression_engine import evaluate, load_model, mine, train
from .progression_stats import compare, scores
from .smollm2_data import freeze


def dispatch_v2(command, cfg):
    if command == "audit":
        from .progression_audit import audit

        return audit(cfg)
    if command == "prepare":
        from .progression_data import prepare

        return prepare(cfg)
    if command in ("run", "public"):
        import fcntl

        with (Path(cfg["root"]) / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return experiment(cfg) if command == "run" else final_evaluation(cfg)
    if command == "report":
        from .progression_v2_report import report

        return report(cfg)
    raise ValueError(command)


def history_rows(folder):
    return [row for file in sorted(folder.glob("*/*.json")) for row in read(file)]


def needs_anchor(history, sft):
    previous_gap = previous_chosen = 0
    consecutive = 0
    for point in history:
        chosen = point["chosen_delta"]
        gap = chosen - point["rejected_delta"]
        weak = point["ifeval"] + point["gsm8k"] <= sft["ifeval"] + sft["gsm8k"]
        consecutive = (
            consecutive + 1
            if gap > previous_gap and chosen < previous_chosen and weak
            else 0
        )
        if consecutive >= 2:
            return True
        previous_chosen, previous_gap = chosen, gap
    return False


def weights(adapter, cfg):
    return (
        sha(Path(adapter) / "adapter_model.safetensors") if adapter else cfg["revision"]
    )


def identity(cfg, directory, adapter, rows):
    freeze(
        directory / "identity.json",
        {
            "protocol": cfg["protocol"],
            "config": digest(cfg),
            "tokenizer": sha(Path(os.environ["PROGRESSION_BASE"]) / "tokenizer.json"),
            "parent": weights(adapter, cfg),
            "rows": digest(rows),
        },
    )


def predict(cfg, adapter, rows, folder):
    identity(cfg, folder, adapter, rows)
    model, tok = load_model(cfg, adapter)
    return evaluate(cfg, model, tok, rows, folder)


def environment(cfg):
    from langdetect import DetectorFactory

    DetectorFactory.seed = cfg["seed"]
    random.seed(cfg["seed"])
    base = Path(os.environ["PROGRESSION_BASE"])
    if (
        sha(base / "model.safetensors")
        != read("docs/repair-results/protocols.json")["base"]["weights"]["base"]
    ):
        raise ValueError("Base weights changed")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA required")
    root = Path(cfg["root"])
    audit = read(root / "audit/verified.json")
    if (
        not audit["passed"]
        or audit["protocol"] != cfg["protocol"]
        or audit["rows"] != digest(read(root / "audit/rows.json"))
    ):
        raise ValueError("Completion audit missing or stale")
    if (
        sha(base / "tokenizer.json")
        != "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"
    ):
        raise ValueError("GPU tokenizer differs from audited tokenizer")
    manifest = read(root / "manifest.json")
    if digest(cfg) != manifest["config"]:
        raise ValueError("Configuration changed")
    return manifest


def load_bank(root, manifest, name):
    rows = read(root / "bank" / (name + ".json"))
    if digest(rows) != manifest["banks"][name]:
        raise ValueError(f"Bank changed: {name}")
    return rows


def train_arm(cfg, stage, parent, rows, dev, folder, baseline=None):
    identity(cfg, folder, parent, rows)
    freeze(folder / "dev-id.json", digest(dev))
    start = time.monotonic()
    best = train(
        cfg,
        stage,
        parent,
        rows,
        dev,
        folder,
        cfg["sft_lr" if stage == "sft" else "dpo_lr"],
        baseline,
    )
    if not (folder / "cost.json").exists():
        freeze(
            folder / "cost.json",
            {
                "wall_seconds_current_invocation": time.monotonic() - start,
                "device": torch.cuda.get_device_name(),
            },
        )
    return best


def trajectory(folder):
    points = sorted(
        folder.glob("step-*/summary.json"),
        key=lambda p: int(p.parent.name.split("-")[1]),
    )
    return [dict(step=int(p.parent.name.split("-")[1]), **read(p)) for p in points]


def experiment(cfg):
    manifest = environment(cfg)
    root, archive, prior = map(Path, (cfg["root"], cfg["archive"], cfg["prior_inputs"]))
    audit = read(root / "audit/verified.json")
    old_parent = archive / "round-1/sft/step-32"
    old_round = archive / "round-2"
    pairs = read(old_round / "mining/pairs.json")
    if digest(pairs) != audit["pairs"] or weights(old_parent, cfg) != audit["parent"]:
        raise ValueError("Ablation inputs changed")
    old_dev = read(prior / "bank/dev.json")
    if digest(old_dev) != read(archive / "manifest.json")["banks"]["dev"]:
        raise ValueError("Historical dev changed")
    replay = root / "ablation"
    train_arm(cfg, "dpo", old_parent, pairs, take(old_dev, 128), replay)
    fixed = history_rows(replay / "step-64/dev")
    old = history_rows(old_round / "dpo/step-64/dev")
    sft_old = history_rows(old_round / "selected-sft-dev")
    freeze(
        replay / "comparison.json",
        {
            "repair": compare(old, fixed, 0),
            "old_dpo": compare(sft_old, old, 0),
            "repaired_dpo": compare(sft_old, fixed, 0),
            "old": trajectory(old_round / "dpo"),
            "new": trajectory(replay),
        },
    )
    dev = load_bank(root, manifest, "dev")
    base_scores = scores(predict(cfg, None, dev, root / "base-dev"))
    sft = train_arm(
        cfg,
        "sft",
        None,
        load_bank(root, manifest, "sft"),
        dev,
        root / "sft",
        base_scores,
    )
    sft_rows = history_rows(sft / "dev")
    sft_scores = scores(sft_rows)
    bank = load_bank(root, manifest, "mine")
    identity(cfg, root / "mining", sft, bank)
    model, tok = load_model(cfg, sft)
    pairs = mine(cfg, model, tok, bank, root / "mining", True)
    del model
    torch.cuda.empty_cache()
    support = Counter(p["task"] for p in pairs)
    freeze(
        root / "mining/support.json",
        {
            "counts": dict(support),
            "sources": {
                t: dict(Counter(p["source"] for p in pairs if p["task"] == t))
                for t in TASKS
            },
        },
    )
    if min(support[t] for t in TASKS) < cfg["min_pairs"]:
        raise ValueError(f"Insufficient verified preferences: {support}")
    arms = {"sft": sft}
    for stage in ("chosen", "dpo"):
        arms[stage] = train_arm(cfg, stage, sft, pairs, dev, root / stage, sft_scores)
    standard = compare(sft_rows, history_rows(arms["dpo"] / "dev"), 1)
    selected = "dpo"
    if not standard["passed"] and needs_anchor(trajectory(root / "dpo"), sft_scores):
        arms["hybrid"] = train_arm(
            cfg, "hybrid", sft, pairs, dev, root / "hybrid", sft_scores
        )
        alt = compare(sft_rows, history_rows(arms["hybrid"] / "dev"), 1)
        if (alt["passed"], alt["pp"]) > (standard["passed"], standard["pp"]):
            selected = "hybrid"
    freeze(
        root / "weights-lock.json",
        {
            "selected": selected,
            "base": cfg["revision"],
            "arms": {
                k: {"path": str(v.relative_to(root).as_posix()), "sha": weights(v, cfg)}
                for k, v in arms.items()
            },
        },
    )


def final_evaluation(cfg):
    manifest = environment(cfg)
    root = Path(cfg["root"])
    lock = read(root / "weights-lock.json")
    arms = {"base": None}
    for k, entry in lock["arms"].items():
        adapter = root / entry["path"]
        if weights(adapter, cfg) != entry["sha"]:
            raise ValueError(f"Locked {k} changed")
        arms[k] = adapter
    for split in ("holdout", "public"):
        if split == "holdout":
            rows = load_bank(root, manifest, split)
        else:
            rows = []
            for i, expected in enumerate(manifest["panels"], 1):
                panel = read(Path(cfg["prior_inputs"]) / f"panels/{i}.json")
                if digest(panel) != expected:
                    raise ValueError("Public panel changed")
                rows.extend(panel)
        names = ["base", "sft", "dpo"] if split == "public" else list(arms)
        if lock["selected"] not in names:
            names.append(lock["selected"])
        results = {k: predict(cfg, arms[k], rows, root / split / k) for k in names}
        pairs = [("base", "sft", 2), ("sft", lock["selected"], 1)]
        if split == "holdout":
            pairs.append(("chosen", "dpo", 0))
        freeze(
            root / split / "results.json",
            {
                "scores": {k: scores(v) for k, v in results.items()},
                "comparisons": {
                    f"{a}-{b}": compare(results[a], results[b], n, True)
                    for a, b, n in pairs
                },
            },
        )
