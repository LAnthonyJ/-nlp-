"""Export measured paired-DPO results and render the two report figures."""

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from .capability import read, save, sha
from .capability_data import digest
from .paired_tasks import KINDS, assess


def render(report, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(report["final"])
    labels = {
        "frozen": "SFT B",
        "previous_dpo": "Previous DPO",
        "sft-42": "Continued SFT",
        "dpo-42": "New DPO",
    }
    fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.4), 4), layout="constrained")
    counts = [report["final"][n]["both"] for n in names]
    percentages = [
        100 * report["final"][n]["both"] / report["final"][n]["n"] for n in names
    ]
    bars = ax.bar(
        [labels.get(n, n) for n in names],
        percentages,
        color=["#60778a" if n != "dpo-42" else "#007b76" for n in names],
    )
    ax.bar_label(
        bars,
        labels=[f"{c}/{report['final'][name]['n']}" for name, c in zip(names, counts)],
        padding=4,
    )
    ax.set(
        ylabel="Both answers correct (%)",
        ylim=(0, max(10, max(percentages) * 1.25)),
        title="Paired generation on unseen task templates",
    )
    fig.savefig(folder / "paired-accuracy.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), layout="constrained", sharex=True)
    for name, run in report["training"].items():
        steps = [row["step"] for row in run["history"]]
        for ax, key, scale in zip(axes, ["margin", "both"], [1, 100 / 30]):
            ax.plot(
                steps,
                [row[key] * scale for row in run["history"]],
                marker="o",
                label=labels.get(name, name),
            )
    axes[0].set(title="Development checkpoints", ylabel="Mean reward margin")
    axes[0].axhline(0, color="grey", linewidth=0.6)
    axes[1].set(
        xlabel="Optimizer update", ylabel="Both answers correct (%)", ylim=(-0.7, None)
    )
    axes[1].set_xticks(
        sorted(
            {h["step"] for run in report["training"].values() for h in run["history"]}
        )
    )
    axes[0].legend()
    fig.savefig(folder / "development.png", dpi=180)
    plt.close(fig)


def export(config, folder):
    folder.mkdir(parents=True, exist_ok=True)
    root = Path(config["output_dir"])
    selection = read(root / "selection.json")
    result = read(root / "results.json")
    result["training"] = selection["training"]
    result["dev_baselines"] = selection["dev_baselines"]
    result["dev_baseline_responses"] = {
        name: read(root / "dev" / f"{name}.json") for name in selection["dev_baselines"]
    }
    result["protocol"] = read(root / "protocol.json")
    pairs = read(root / "pairs.json")
    result["mining"] = {}
    attempts = []
    tasks = {row["id"]: row for row in read(root / "suite.json")["train"]}
    for kind in KINDS:
        mined = [
            r
            for path in sorted((root / "mining").glob(kind + "-*.json"))
            for r in read(path)
        ]
        chosen = [r for r in pairs if r["kind"] == kind]
        result["mining"][kind] = dict(
            prompts=len(mined),
            eligible=sum(bool(r["pair"]) for r in mined),
            selected=len(chosen),
            reasons=dict(Counter(r["reason"] for r in mined)),
            source=dict(Counter(r["chosen_source"] for r in chosen)),
            answer_counts=dict(Counter(r["answer"] for r in chosen)),
        )
        attempts.extend(mined)
    result["mining_audit_hash"] = digest(attempts)
    audit = []
    for row in attempts:
        task = tasks[row["id"]]
        candidates = [
            dict(**candidate, **assess(task, **candidate))
            for candidate in row["candidates"]
        ]
        audit.append(dict(**task, candidates=candidates, reason=row["reason"]))
    raw = json.dumps(audit, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    (folder / "mining.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    result["pairs_hash"] = digest(pairs)
    save(folder / "pairs.json", pairs)
    result["dev_checkpoints"] = {}
    result["weights"] = {}
    result["regression"] = {}
    result["regression_protocols"] = {}
    outcomes = {}
    regression_rows = {}
    for name, adapter in selection["arms"].items():
        result["weights"][name] = sha(Path(adapter) / "adapter_model.safetensors")
        outcomes[name] = read(root / "holdout" / f"{name}.json")
        scores = read(root / "regression" / name / "scores.json")
        result["regression_protocols"][name] = read(
            root / "regression" / name / "protocol.json"
        )
        regression_rows[name] = [
            dict(**a, score=b)
            for a, b in zip(read(root / "regression" / name / "responses.json"), scores)
        ]
        if len(regression_rows[name]) != 55 or any(
            row["task_id"] != row["score"]["task_id"] for row in regression_rows[name]
        ):
            raise ValueError("Regression responses and scores do not match")
        result["regression"][name] = {
            kind: sum(r["correct"] for r in scores if r["category"] == kind)
            for kind in ("math", "code")
        }
        if name in selection["training"]:
            result["dev_checkpoints"][name] = [
                dict(
                    step=h["step"],
                    responses=read(root / name / f"dev-{h['step']}.json"),
                )
                for h in selection["training"][name]["history"]
            ]
    save(folder / "generations.json", outcomes)
    save(folder / "regression.json", regression_rows)
    save(folder / "metrics.json", result)
    render(result, folder)
    files = [
        "metrics.json",
        "pairs.json",
        "generations.json",
        "regression.json",
        "mining.json.gz",
        "paired-accuracy.png",
        "development.png",
    ]
    save(folder / "manifest.json", {name: sha(folder / name) for name in files})
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument(
        "--metrics", type=Path, help="Redraw the figures from exported measurements"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.metrics:
        args.output.mkdir(parents=True, exist_ok=True)
        render(read(args.metrics), args.output)
    else:
        export(read(args.config), args.output)
