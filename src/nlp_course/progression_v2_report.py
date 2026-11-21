"""Independently score returned rows and publish a reproducible v2 report."""

import gzip
import json
import os
from pathlib import Path

from .capability import read, save, sha
from .capability_data import digest
from .progression_data import TASKS
from .progression_engine import grade
from .progression_stats import compare, scores
from .progression_v2 import history_rows, load_bank, trajectory


def history_rows_for_task(folder):
    return [r for p in sorted(folder.glob("*.json")) for r in read(p)]


def report(cfg):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from langdetect import DetectorFactory

    DetectorFactory.seed = cfg["seed"]
    root = Path(os.getenv("PROGRESSION_RETURN", cfg["root"]))
    manifest = read(root / "manifest.json")
    if manifest != read(Path(cfg["root"]) / "manifest.json") or manifest[
        "config"
    ] != digest(cfg):
        raise ValueError("Configuration or manifest changed")
    lock = read(root / "weights-lock.json")
    for arm in lock["arms"].values():
        if sha(root / arm["path"] / "adapter_model.safetensors") != arm["sha"]:
            raise ValueError("Returned weights changed")
    selected = lock["selected"]
    predictions, metrics, cases = [], {}, {}
    for split in ("holdout", "public"):
        if split == "holdout":
            questions = load_bank(root, manifest, split)
        else:
            questions = []
            for i, expected in enumerate(manifest["panels"], 1):
                panel = read(Path(cfg["prior_inputs"]) / f"panels/{i}.json")
                if digest(panel) != expected:
                    raise ValueError("Public dataset changed")
                questions.extend(panel)
        index = {r["id"]: r for r in questions}
        saved = read(root / split / "results.json")
        results = {}
        for arm, summary in saved["scores"].items():
            rows = history_rows(root / split / arm)
            if set(index) != {r["id"] for r in rows} or summary != scores(rows):
                raise ValueError("Incomplete or inconsistent returned predictions")
            for r in rows:
                q = index[r["id"]]
                if "choices" in q:
                    vals = [
                        v / len(a.removeprefix(" "))
                        for v, a in zip(r["logprob"], q["choices"], strict=True)
                    ]
                    correct = max(range(len(vals)), key=vals.__getitem__) == q["gold"]
                else:
                    correct = grade(q, r["response"])
                if r["task"] != q["task"] or bool(correct) != r["correct"]:
                    raise ValueError(f"Score mismatch: {q['id']}")
                predictions.append(
                    dict(split=split, arm=arm, dataset=q.get("dataset", q["task"]), **r)
                )
            results[arm] = rows
        for key, expected in saved["comparisons"].items():
            a, b = key.split("-")
            target = 2 if a == "base" else 1 if a == "sft" else 0
            if compare(results[a], results[b], target, True) != expected:
                raise ValueError("Paired interval mismatch")
        metrics[split] = saved
        before = {r["id"]: r for r in results["sft"]}
        for r in results[selected]:
            old = before[r["id"]]
            if r["correct"] != old["correct"]:
                kind = "improved" if r["correct"] else "regressed"
                cases.setdefault(
                    split + "-" + kind,
                    {"question": index[r["id"]], "sft": old, "dpo": r},
                )
    ablation = read(root / "ablation/comparison.json")
    out = Path("docs/progression-v2-results")
    out.mkdir(parents=True, exist_ok=True)
    costs = {p.parent.name: read(p) for p in root.glob("*/cost.json")}
    audit = read(root / "audit/verified.json")
    support = read(root / "mining/support.json")
    for task in TASKS:
        candidates = history_rows_for_task(root / "mining" / task)
        selected_pairs = [
            r for r in read(root / "mining/pairs.json") if r["task"] == task
        ]
        support.setdefault("collection", {})[task] = {
            "attempted": len(candidates),
            "eligible": sum(bool(r["pair"]) for r in candidates),
            "selected": len(selected_pairs),
            "selected_per_attempt": len(selected_pairs) / len(candidates),
            "greedy_errors": sum(r["greedy_error"] for r in selected_pairs),
        }
    identity_files = list(root.rglob("identity.json"))
    finished_files = [root / "public/results.json", root / "holdout/results.json"]
    costs["single_gpu_wall_seconds"] = max(
        p.stat().st_mtime for p in finished_files
    ) - min(p.stat().st_mtime for p in identity_files)
    save(
        out / "metrics.json",
        {
            "metrics": metrics,
            "costs": costs,
            "ablation": ablation,
            "training_history": {arm: trajectory(root / arm) for arm in lock["arms"]},
            "preference_hash": digest(read(root / "mining/pairs.json")),
            "audit": audit,
            "support": support,
            "inputs": manifest,
            "weights": lock,
        },
    )
    save(out / "cases.json", cases)
    with gzip.open(out / "predictions.jsonl.gz", "wt", encoding="utf8") as stream:
        for r in predictions:
            stream.write(json.dumps(r, ensure_ascii=False) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4))
    for task in TASKS:
        ax.plot(
            ["Base", "SFT", selected.upper()],
            [
                metrics["public"]["scores"][arm][task]
                for arm in ("base", "sft", selected)
            ],
            marker="o",
            label=task,
        )
    ax.set_ylabel("Public accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "public.png")
    plt.close(fig)
    points = trajectory(root / "dpo")
    steps = [r["step"] for r in points]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4))
    left.plot(
        steps, [r["chosen_delta"] - r["rejected_delta"] for r in points], marker="o"
    )
    left.set_ylabel("Preference margin change (nats)")
    for task in ("ifeval", "gsm8k"):
        right.plot(steps, [r[task] for r in points], marker="o", label=task)
    right.set_ylabel("Development accuracy (%)")
    right.legend()
    for ax in (left, right):
        ax.set_xlabel("Optimizer updates")
    fig.tight_layout()
    fig.savefig(out / "diagnostics.png")
    plt.close(fig)
    lines = [
        "# Completion-boundary ablation and one-stage DPO",
        "",
        "[Reproduction](../progression-v2.md) · [Metrics, costs and hashes](metrics.json) · [Changed answers](cases.json)",
        "",
    ]
    for split, values in metrics.items():
        math = "SVAMP" if split == "holdout" else "GSM8K"
        instruction = "Custom instructions" if split == "holdout" else "IFEval strict"
        lines.extend(
            [
                f"## {split.title()}",
                "",
                f"| Model | PIQA | HellaSwag | {instruction} | {math} | Mean |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for arm, v in values["scores"].items():
            lines.append(
                "| "
                + arm
                + " | "
                + " | ".join(f"{v[k]:.2f}%" for k in (*TASKS, "mean"))
                + " |"
            )
        lines.append("")
        for key, v in values["comparisons"].items():
            lo, hi = v["ci95"]
            lines.append(
                f"- {key} change {v['pp']:+.2f} pp (95% CI {lo:+.2f} to {hi:+.2f}), adjusted lower bound {v['lower']:+.2f}."
            )
        lines.append("")
    formal = metrics["holdout"]["comparisons"]
    passed = formal["base-sft"]["passed"] and formal["sft-" + selected]["passed"]
    evidence = formal["chosen-dpo"]["ci95"][0] > 0
    lines.extend(
        [
            f"Single-seed prospective gate: {'passed' if passed else 'missed'}.",
            f"Incremental benefit of standard DPO over chosen-only training: {'supported on this holdout' if evidence else 'not established'}.",
            "",
            f"Legacy step-64 repair contrast: {ablation['repair']['pp']:+.2f} pp on the original development set.",
            "",
            "![Public task results](public.png)",
            "",
            "![Preference and generation trajectories](diagnostics.png)",
        ]
    )
    lines.extend(
        [
            "",
            "## Attribution and training cost",
            "",
            "The archived comparator was trained on an RTX 5090. The replay uses the same parent, examples, order and training settings on the available GPU.",
            "",
            "| Task | Old wrong to new correct | Old correct to new wrong | Change (pp) |",
            "|---|---:|---:|---:|",
        ]
    )
    for task, result in ablation["repair"]["tasks"].items():
        lines.append(
            f"| {task} | {result['gained']} | {result['lost']} | {result['pp']:+.2f} |"
        )
    lines.extend(
        [
            "",
            "| Training arm | GPU | Elapsed minutes including checkpoint scoring |",
            "|---|---|---:|",
        ]
    )
    for name in ("ablation", "sft", "dpo", "chosen", "hybrid"):
        if name in costs:
            lines.append(
                f"| {name} | {costs[name]['device']} | {costs[name]['wall_seconds_current_invocation'] / 60:.1f} |"
            )
    lines.extend(
        [
            "",
            "The prospective instruction task tests sorting and filtering under two output formats. The public suite uses original IFEval prompts. The public dataset was measured in a previous experiment; the confidence intervals condition on this seed and model.",
        ]
    )
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf8")
