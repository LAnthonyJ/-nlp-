"""Export the observed three-stage results and optional locked full-suite scores."""

import gzip
import json
import random
from pathlib import Path

from .capability import read, save, sha
from .capability_data import digest
from .progression_data import TASKS
from .progression_stats import compare, scores
from .smollm2_data import freeze


def verify_return(root, original):
    """Recompute public scores, intervals and adapter hashes after cloud transfer."""
    from .progression_data import final_value
    from .evaluation import parse_number
    from .smollm2_eval import score_instruction
    from langdetect import DetectorFactory

    DetectorFactory.seed = 42
    random.seed(42)
    manifest = read(original / "manifest.json")
    if read(root / "manifest.json") != manifest:
        raise ValueError("Returned manifest differs from the frozen local data")
    questions = {}
    for i, expected in enumerate(manifest["panels"], 1):
        panel = read(original / f"panels/{i}.json")
        if digest(panel) != expected:
            raise ValueError("Changed local panel")
        for row in panel:
            if row["id"] in questions:
                raise ValueError("Duplicate panel ID")
            questions[row["id"]] = row
    full = read(root / "public.json")

    def checked_rows(folder, name, expected, summary):
        items = []
        for p in sorted(folder.glob("*/*.json")):
            for r in read(p):
                q = questions[r["id"]]
                if "choices" in q:
                    values = [
                        v / len(a.removeprefix(" "))
                        for v, a in zip(r["logprob"], q["choices"], strict=True)
                    ]
                    correct = (
                        max(range(len(values)), key=values.__getitem__) == q["gold"]
                    )
                    if values != r["normalized"]:
                        raise ValueError("Candidate normalization changed")
                elif q["task"] == "gsm8k":
                    correct = final_value(r["response"]) == parse_number(q["gold"])
                else:
                    correct = bool(
                        score_instruction(q, r["response"])["prompt_level_strict_acc"]
                    )
                if correct != r["correct"] or q["task"] != r["task"]:
                    raise ValueError(f"Public score mismatch: {name} {r['id']}")
                items.append(r)
        if {r["id"] for r in items} != expected or scores(items) != summary:
            raise ValueError("Incomplete public evaluation")
        return items

    arms = {
        name: checked_rows(
            root / "public" / name, name, set(questions), full["scores"][name]
        )
        for name in ("base", "sft", "dpo")
    }
    for a, b, target in [("base", "sft", 2), ("sft", "dpo", 1)]:
        if compare(arms[a], arms[b], target, True) != full[b]:
            raise ValueError("Paired comparison mismatch")
    final = read(root / "finished.json")
    locks = read(root / "public-lock.json")
    complete = [
        read(root / f"round-{i}/outcome.json") for i in range(1, final["attempts"] + 1)
    ]
    for i, run in enumerate(complete, 1):
        if "dpo" not in run:
            continue
        folder = root / f"round-{i}"
        if run["opened"]:
            panel = read(original / f"panels/{run['panel']}.json")
            expected = {r["id"] for r in panel}
            stages = {
                name: checked_rows(
                    folder / f"formal-{name}",
                    name,
                    expected,
                    run["formal_scores"][name],
                )
                for name in ("base", "sft", "dpo")
            }
            gates = []
            for before, after, target in (("base", "sft", 2), ("sft", "dpo", 1)):
                delta = compare(stages[before], stages[after], target, True)
                if delta != run[f"formal_{after}"]:
                    raise ValueError("Sealed-panel inference mismatch")
                gates.append(delta["passed"])
            if all(gates) != run["passed"]:
                raise ValueError("Formal acceptance mismatch")
        pairs = read(folder / "mining/pairs.json")
        random.Random(read(root / "environment.json")["seed"]).shuffle(pairs)
        parent = (
            root / Path(run["sft"]).relative_to(original) / "adapter_model.safetensors"
        )
        cache = read(folder / "dpo/reference.json")
        if cache["key"] != digest([sha(parent), pairs]) or len(cache["values"]) != len(
            pairs
        ):
            raise ValueError(
                "DPO reference does not match its frozen SFT and ordered preferences"
            )
    last = next(r for r in reversed(complete) if "dpo" in r)
    for stage in ("sft", "dpo"):
        path = Path(last[stage]).relative_to(original)
        if sha(root / path / "adapter_model.safetensors") != locks[stage]:
            raise ValueError("Final adapter hash mismatch")
    print(
        f"Verified {sum(map(len, arms.values())):,} predictions and both selected adapters",
        flush=True,
    )


def reuse_formal_batches(root, rows, size, weights):
    """Reuse an identical locked model AND an identical ordered inference batch."""
    for claim in (root / "opened").glob("*.json"):
        lock = read(claim)
        folder = root / f"round-{lock['attempt']}"
        for stage, identity in weights.items():
            if lock[stage] != identity:
                continue
            for task in TASKS:
                ids = [r["id"] for r in rows if r["task"] == task]
                offsets = {
                    tuple(ids[i : i + size]): i for i in range(0, len(ids), size)
                }
                for p in (folder / f"formal-{stage}" / task).glob("*.json"):
                    predictions = read(p)
                    key = tuple(r["id"] for r in predictions)
                    if key in offsets:
                        freeze(
                            root / "public" / stage / task / f"{offsets[key]}.json",
                            predictions,
                        )


def public(cfg):
    import gc
    import torch
    from langdetect import DetectorFactory
    from .progression_engine import evaluate, load_model

    DetectorFactory.seed = cfg["seed"]
    random.seed(cfg["seed"])
    root = Path(cfg["root"])
    final = read(root / "finished.json")
    history = [
        read(root / f"round-{i}/outcome.json") for i in range(1, final["attempts"] + 1)
    ]
    trained = [r for r in history if "dpo" in r]
    if not trained:
        raise ValueError("No trained DPO available")
    # Lock the chronological last complete chain; no selection on public scores.
    record = trained[-1]
    manifest = read(root / "manifest.json")
    rows = []
    for i, checksum in enumerate(manifest["panels"], 1):
        panel = read(root / f"panels/{i}.json")
        if digest(panel) != checksum:
            raise ValueError("Public panel hash mismatch")
        rows.extend(panel)
    models = {"base": None, "sft": record["sft"], "dpo": record["dpo"]}
    freeze(
        root / "public-lock.json",
        {
            k: sha(Path(v) / "adapter_model.safetensors") if v else cfg["revision"]
            for k, v in models.items()
        },
    )
    all_rows = {}
    reuse_formal_batches(root, rows, cfg["eval_batch"], read(root / "public-lock.json"))
    for name, path in models.items():
        model, tok = load_model(cfg, path)
        print("Locked public evaluation:", name, flush=True)
        all_rows[name] = evaluate(cfg, model, tok, rows, root / "public" / name)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    freeze(
        root / "public.json",
        dict(
            scores={k: scores(v) for k, v in all_rows.items()},
            sft=compare(all_rows["base"], all_rows["sft"], 2, True),
            dpo=compare(all_rows["sft"], all_rows["dpo"], 1, True),
            counts={t: sum(r["task"] == t for r in rows) for t in TASKS},
        ),
    )


def report(root, dest=Path("docs/progression-results")):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    final = read(root / "finished.json")
    full = read(root / "public.json")
    history = [
        read(root / f"round-{i}/outcome.json") for i in range(1, final["attempts"] + 1)
    ]
    dest.mkdir(parents=True, exist_ok=True)
    audit = []
    for i in range(1, final["attempts"] + 1):
        folder = root / f"round-{i}"
        stages = {}
        for stage in ("sft", "dpo"):
            stages[stage] = {
                p.parent.name: read(p)
                for p in sorted((folder / stage).glob("step-*/summary.json"))
            }
        collection = {}
        for task in TASKS:
            sampled = [
                r
                for file in (folder / "mining" / task).glob("*.json")
                for r in read(file)
            ]
            collection[task] = dict(
                attempted=len(sampled),
                eligible=sum(bool(r["pair"]) for r in sampled),
                retained=history[i - 1]["pairs"][task],
            )
        audit.append(
            dict(
                design=read(folder / "design.json"),
                wall_seconds=(folder / "outcome.json").stat().st_mtime
                - (folder / "design.json").stat().st_mtime,
                checkpoints=stages,
                mining=collection,
            )
        )
    save(
        dest / "metrics.json",
        dict(
            public=full,
            history=history,
            audit=audit,
            acceptance=final,
            weights=read(root / "public-lock.json"),
            data=read(root / "manifest.json"),
            environment=read(root / "environment.json"),
        ),
    )
    arms = ["base", "sft", "dpo"]
    colors = ["#74899b", "#315a92", "#097a66"]
    fig, (l, r) = plt.subplots(1, 2, figsize=(9, 4.0), layout="constrained")
    means = [full["scores"][m]["mean"] for m in arms]
    bars = l.bar(["Base", "SFT", "DPO"], means, color=colors)
    l.bar_label(bars, fmt="%.2f")
    l.set(ylim=(0, 100), ylabel="Equal-weight accuracy (%)", title="Four public tasks")
    for i, m in enumerate(arms[1:]):
        value = full[m]
        r.hlines(i, *value["ci95"], colors=colors[i + 1])
        r.plot(value["pp"], i, "o", color=colors[i + 1])
    r.set(
        yticks=[0, 1],
        yticklabels=["SFT minus Base", "DPO minus SFT"],
        xlabel="Difference (percentage points)",
        title="Paired 95% intervals",
    )
    r.axvline(0, color="gray", ls="--")
    fig.savefig(dest / "progression.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for i, m in enumerate(arms):
        ax.bar(
            [k + (i - 1) * 0.25 for k in range(4)],
            [full["scores"][m][t] for t in TASKS],
            0.25,
            label=m.upper(),
            color=colors[i],
        )
    ax.set(xticks=range(4), xticklabels=TASKS, ylim=(0, 100), ylabel="Accuracy (%)")
    ax.legend()
    fig.savefig(dest / "tasks.png", dpi=160)
    plt.close(fig)
    records = {}
    for arm in arms:
        records[arm] = {}
        for p in sorted((root / "public" / arm).glob("*/*.json")):
            for row in read(p):
                records[arm][row["id"]] = row
    cases = {}
    for a, b in zip(arms, arms[1:]):
        cases[b] = {}
        for direction in ("gain", "loss"):
            sample = next(
                (
                    k
                    for k in sorted(records[b])
                    if records[b][k]["correct"] != records[a][k]["correct"]
                    and records[b][k]["correct"] == (direction == "gain")
                ),
                None,
            )
            cases[b][direction] = (
                dict(before=records[a][sample], after=records[b][sample])
                if sample
                else None
            )
    save(dest / "cases.json", cases)
    with gzip.open(dest / "predictions.jsonl.gz", "wt", encoding="utf8") as f:
        for arm, items in records.items():
            for row in items.values():
                f.write(json.dumps(dict(model=arm, **row), ensure_ascii=False) + "\n")
    lines = [
        "# Qwen2.5-0.5B public evaluation after one SFT stage and one round of DPO",
        "",
        f"Across {final['attempts']} attempts, the prespecified holdout criteria were {'met' if final['passed'] else 'not met'}. After checkpoint selection, each stage was evaluated on {sum(full['counts'].values()):,} questions with equal weight for each of the four tasks. DPO started from the selected SFT checkpoint.",
        "",
        "| Stage | PIQA | HellaSwag | IFEval strict prompt-level | GSM8K | Equal-weight mean |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in arms:
        lines.append(
            "| "
            + m.upper()
            + " | "
            + " | ".join(f"{full['scores'][m][k]:.2f}%" for k in (*TASKS, "mean"))
            + " |"
        )
    lines.extend(
        [
            "",
            "## Prespecified holdout evaluation",
            "",
            "| Attempt and panel | Base mean | SFT mean | DPO mean | SFT gain / adjusted one-sided lower bound | DPO gain / adjusted one-sided lower bound |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for i, r in enumerate(history, 1):
        if r["opened"]:
            formal = r["formal_scores"]
            a, b = r["formal_sft"], r["formal_dpo"]
            lines.append(
                f"| Attempt {i} / panel {r['panel']} | {formal['base']['mean']:.2f}% | {formal['sft']['mean']:.2f}% | {formal['dpo']['mean']:.2f}% | {a['pp']:+.2f} / {a['lower']:+.3f} | {b['pp']:+.2f} / {b['lower']:+.3f} |"
            )
    lines.extend(
        [
            "",
            "The sealed panel determines acceptance. The full benchmark includes the opened questions and describes the selected checkpoints after training. Gains and lower bounds are in percentage points.",
            "",
            "## Changes on the full benchmark",
            "",
            "| Stage | Mean change (pp) | Paired 95% interval | Largest task regression (pp) |",
            "|---|---:|---:|---|",
        ]
    )
    for name in ("sft", "dpo"):
        change = full[name]
        worst = min(TASKS, key=lambda t: change["tasks"][t]["pp"])
        lo, hi = change["ci95"]
        lines.append(
            f"| {name.upper()} | {change['pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {worst} {change['tasks'][worst]['pp']:+.2f} |"
        )
    if full["dpo"]["ci95"][0] <= 0 <= full["dpo"]["ci95"][1]:
        lines.extend(
            [
                "",
                f"The full-benchmark DPO gain over SFT was {full['dpo']['pp']:+.2f} percentage points. Its paired 95% interval includes zero, so the larger evaluation does not confirm the gain observed on the sealed panel.",
            ]
        )
    for stage in ("sft", "dpo"):
        for task, delta in full[stage]["tasks"].items():
            if delta["pp"] < -2:
                lines.append(
                    f"On the full {task} benchmark, {stage.upper()} fell {-delta['pp']:.2f} percentage points from the previous stage, exceeding the two-point regression threshold."
                )
    lines.extend(
        [
            "",
            "## Training attempts",
            "",
            "| Attempt | SFT development gain | DPO development gain | Panel opened after selection | Single-GPU minutes |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for i, r in enumerate(history, 1):
        d = f"{r['dev_dpo']['pp']:+.2f}" if "dev_dpo" in r else "Insufficient preference pairs"
        lines.append(
            f"| {i} | {r['dev_sft']['pp']:+.2f} | {d} | {r.get('panel', '—')} | {audit[i - 1]['wall_seconds'] / 60:.1f} |"
        )
    lines.append("")
    for i, r in enumerate(history, 1):
        if r["passed"]:
            lines.append(f"Both gains met the sealed-panel criteria on attempt {i}.")
            continue
        if "dev_dpo" not in r:
            lines.append(
                f"On attempt {i}, at least one task yielded fewer than 128 eligible preference pairs. See the metrics file for counts."
            )
            continue
        phase = "holdout" if r["opened"] else "development"
        comparisons = (
            [r["formal_sft"], r["formal_dpo"]]
            if r["opened"]
            else [r["dev_sft"], r["dev_dpo"]]
        )
        for stage, delta in zip(("SFT", "DPO"), comparisons, strict=True):
            if delta["passed"]:
                continue
            worst = min(TASKS, key=lambda k: delta["tasks"][k]["pp"])
            task = delta["tasks"][worst]
            details = []
            if delta["pp"] < (2 if stage == "SFT" else 1):
                details.append("The overall gain was below the prespecified threshold")
            if task["pp"] < -2:
                details.append(
                    f"{worst} gained {task['gained']} correct responses and lost {task['lost']}, a decline of {-task['pp']:.2f} percentage points"
                )
            if delta.get("lower", 1) <= 0:
                details.append(
                    f"The adjusted one-sided lower bound was {delta['lower']:.2f} percentage points"
                )
            lines.append(
                f"On attempt {i}, the {phase} change for {stage} relative to the previous stage was {delta['pp']:+.2f} percentage points."
                + " ".join(details)
                + "."
            )
        if i < len(history):
            design = audit[i]["design"]
            prior = audit[i - 1]["design"]
            if design["dev"] > prior["dev"]:
                lines.append(
                    f"The next attempt increased the development set to {design['dev']} examples per task and retrained from Base."
                )
            elif design["hard"] and not prior["hard"]:
                lines.append(
                    "The next attempt started from the same SFT checkpoint and prioritized prompts with an incorrect first response and a correct alternative."
                )
            elif design["dpo_lr"] < prior["dpo_lr"]:
                lines.append(f"The next attempt reduced the DPO learning rate to {design['dpo_lr']:g}.")
    lines.extend(
        [
            "",
            "Development gains are in percentage points. Instruction development uses independent exercises. The sealed and full public evaluations use original IFEval. Timing includes development evaluation and preference collection. Metrics include configurations, paired gains and losses, and changes in answer probabilities.",
            f"On {read(root / 'environment.json')['gpu']}, environment verification through final result writing took {((root / 'public.json').stat().st_mtime - (root / 'environment.json').stat().st_mtime) / 3600:.2f} hours including waiting and scoring.",
            "",
            "[Training and reproduction](../progression.md)",
        ]
    )
    lines.extend(
        [
            "",
            "![Overall scores and paired intervals](progression.png)",
            "",
            "![Scores by task](tasks.png)",
            "",
            "The records include unsuccessful attempts, acceptance decisions, holdout hashes, individual outputs and examples of gains and regressions. Historical exposure to public questions limits conclusions to this seed, these checkpoints and this protocol.",
            "",
            "[Metrics and acceptance](metrics.json) · [Per-question results](predictions.jsonl.gz) · [Examples](cases.json)",
        ]
    )
    (dest / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    save(
        dest / "manifest.json",
        {
            name: sha(dest / name)
            for name in (
                "metrics.json",
                "cases.json",
                "predictions.jsonl.gz",
                "progression.png",
                "tasks.png",
            )
        },
    )
