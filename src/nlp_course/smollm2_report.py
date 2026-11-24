"""Recompute the public result tables, paired intervals and figures from predictions."""

import gzip
import json
import argparse
import os
import random
from pathlib import Path
import sqlite3

import numpy as np

from .capability import read, save, sha
from .capability_data import digest
from .smollm2_data import assets
from .smollm2_eval import ordered

NAMES = {
    "base": "Qwen Base",
    "instruct": "Qwen Instruct",
    "frozen": "Frozen SFT B",
    "previous_dpo": "Previous DPO",
    "sft-42": "Continued SFT",
    "dpo-42": "Upgraded DPO",
}
CONTROLS = ("sft-42", "frozen", "previous_dpo")
IF_KEYS = (
    "prompt_level_strict_acc",
    "inst_level_strict_acc",
    "prompt_level_loose_acc",
    "inst_level_loose_acc",
)


def verify_row(task, r, src):
    if r["id"] != src["id"]:
        raise ValueError("Source ID mismatch")
    if task == "ifeval":
        s = r["scores"]
        for mode in ("strict", "loose"):
            checks = s[f"inst_level_{mode}_acc"]
            if len(checks) != len(src["instructions"]) or bool(
                s[f"prompt_level_{mode}_acc"]
            ) != all(checks):
                raise ValueError("Invalid instruction score aggregation")
        correct = bool(s["prompt_level_strict_acc"])
    else:
        if len(r["logprob"]) != len(src["choices"]) or src["gold"] != r["gold"]:
            raise ValueError("Candidate alignment or gold mismatch")
        normalized = [
            lp / (len(choice) - int(choice.startswith(" ")))
            for lp, choice in zip(r["logprob"], src["choices"], strict=True)
        ]
        if normalized != r["normalized"] or int(np.argmax(normalized)) != r["pred"]:
            raise ValueError("Cannot reproduce candidate ranking")
        correct = r["pred"] == src["gold"]
    if correct != r["correct"]:
        raise ValueError("Cannot reproduce recorded outcome")


def units(rows, key):
    if key == "accuracy":
        return np.array([(int(r["correct"]), 1) for r in rows])
    if key.startswith("inst_"):
        return np.array([(sum(r["scores"][key]), len(r["scores"][key])) for r in rows])
    return np.array([(int(r["scores"][key]), 1) for r in rows])


def paired(a, b, key, seed):
    if [r["id"] for r in a] != [r["id"] for r in b]:
        raise ValueError("Paired comparison requires identical ordered IDs")
    left, right = units(a, key), units(b, key)
    if not np.array_equal(left[:, 1], right[:, 1]):
        raise ValueError("A paired prompt has different numbers of instructions")
    diff = left[:, 0] - right[:, 0]
    # Sampling multinomial counts of identical (delta, denominator) cells is
    # equivalent to drawing full prompts with replacement; instruction checks stay together.
    cells, counts = np.unique(
        np.column_stack((diff, left[:, 1])), axis=0, return_counts=True
    )
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(a), counts / len(a), size=10000)
    interval = np.percentile(
        100 * (draws @ cells[:, 0]) / (draws @ cells[:, 1]), [2.5, 97.5]
    )
    return dict(
        gained=int(diff.clip(min=0).sum()),
        lost=int((-diff).clip(min=0).sum()),
        delta_pp=float(100 * diff.sum() / left[:, 1].sum()),
        ci95=interval.tolist(),
    )


def figures(folder, metrics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = list(metrics["tasks"])
    colors = ["#8599a3", "#7994ae", "#547181", "#676381", "#365d9b", "#00836c"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained", sharey=True)
    for ax, task in zip(axes, tasks):
        arms = metrics["tasks"][task]
        bars = ax.bar(
            range(len(arms)),
            [arms[n]["percent"] for n in arms],
            color=[colors[list(NAMES).index(n)] for n in arms],
        )
        ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set_xticks(
            range(len(arms)),
            [NAMES[n] for n in arms],
            rotation=48,
            ha="right",
            fontsize=8,
        )
        ax.set_title(task.upper() if task != "hellaswag" else "HellaSwag")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)")
    fig.savefig(folder / "accuracy.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 3.3), layout="constrained")
    for i, task in enumerate(tasks):
        r = metrics["comparisons"][task]["sft-42"]["primary"]
        ax.plot(r["ci95"], [i, i], lw=3, color="#348179")
        ax.scatter([r["delta_pp"]], [i], color="#124c4a", zorder=2)
    ax.axvline(0, color="gray", linestyle="--", lw=1)
    ax.set_yticks(range(len(tasks)), ["PIQA", "HellaSwag", "IFEval strict prompt"])
    ax.invert_yaxis()
    ax.set_xlabel("Upgraded DPO minus continued SFT (percentage points)")
    ax.set_title("Paired differences with 95% bootstrap intervals")
    fig.savefig(folder / "paired-differences.png", dpi=170)
    plt.close(fig)


def export(cfg, folder):
    root = Path(cfg["root"])
    results = dict(
        tasks={},
        comparisons={},
        cases={},
        checks={},
        seed=cfg["seed"],
        bootstrap_draws=10000,
    )
    protocols = {}
    corpus = {}
    with (
        sqlite3.connect(
            f"file:{(root / 'run.sqlite').resolve().as_posix()}?mode=ro", uri=True
        ) as db,
        sqlite3.connect(
            f"file:{(root / 'check.sqlite').resolve().as_posix()}?mode=ro", uri=True
        ) as smoke,
    ):
        for task, settings in cfg["tasks"].items():
            src = read(root / f"{task}.json")
            n = settings["n"]
            if len(src) != n or len({r["id"] for r in src}) != n:
                raise ValueError("Incorrect frozen split")
            data_hash = digest(src)
            arms = {}
            results["tasks"][task] = {}
            protocols[task] = {}
            results["checks"][task] = {}
            for name in NAMES:
                if task == "ifeval" and name == "base":
                    continue
                proto = read(root / "run" / task / f"{name}.json")
                key = proto.pop("key")
                if digest(proto) != key or proto["data"] != data_hash:
                    raise ValueError("Changed result protocol")
                rows = ordered(db, key, src)
                for row, source in zip(rows, src, strict=True):
                    verify_row(task, row, source)
                cost = db.execute(
                    "SELECT sum(n),sum(seconds),max(peak) FROM costs WHERE protocol=?",
                    (key,),
                ).fetchone()
                if cost[0] != n:
                    raise ValueError("Incomplete timing log")
                probe = read(root / "check" / task / f"{name}.json")
                probes = ordered(smoke, probe["key"], src[:8])
                for p, s in zip(probes, src[:8], strict=True):
                    verify_row(task, p, s)
                seconds, peak = smoke.execute(
                    "SELECT sum(seconds),max(peak) FROM costs WHERE protocol=?",
                    (probe["key"],),
                ).fetchone()
                results["checks"][task][name] = dict(
                    n=len(probes),
                    seconds=seconds,
                    peak_mib=peak / 2**20,
                    prompts_per_second=len(probes) / seconds,
                    projected_minutes_at_probe_batch=n * seconds / len(probes) / 60,
                    formal_first8_matches=sum(
                        p["correct"] == r["correct"]
                        for p, r in zip(probes, rows[:8], strict=True)
                    ),
                )
                arms[name] = rows
                count = sum(int(r["correct"]) for r in rows)
                item = dict(
                    correct=count,
                    total=n,
                    percent=100 * count / n,
                    seconds=cost[1],
                    peak_mib=cost[2] / 2**20,
                )
                if task == "ifeval":
                    item["truncated"] = sum(r["truncated"] for r in rows)
                    for metric in IF_KEYS:
                        sums = units(rows, metric).sum(axis=0)
                        item[metric] = dict(
                            passed=int(sums[0]),
                            total=int(sums[1]),
                            percent=float(100 * sums[0] / sums[1]),
                        )
                results["tasks"][task][name] = item
                protocols[task][name] = dict(key=key, **proto)
            key = "prompt_level_strict_acc" if task == "ifeval" else "accuracy"
            results["comparisons"][task] = {}
            for control in CONTROLS:
                comparison = {
                    "primary": paired(arms["dpo-42"], arms[control], key, cfg["seed"])
                }
                if task == "ifeval":
                    comparison.update(
                        {
                            k: paired(arms["dpo-42"], arms[control], k, cfg["seed"])
                            for k in IF_KEYS
                        }
                    )
                results["comparisons"][task][control] = comparison
            cases = {}
            for label, wins in [("gain", True), ("loss", False)]:
                cases[label] = next(
                    (
                        dict(
                            id=a["id"],
                            instructions=s.get("instructions"),
                            continued_sft=b,
                            upgraded_dpo=a,
                        )
                        for a, b, s in zip(
                            arms["dpo-42"], arms["sft-42"], src, strict=True
                        )
                        if a["correct"] == wins and b["correct"] != wins
                    ),
                    None,
                )
            results["cases"][task] = cases
            corpus[task] = arms
    results["calibration"] = {}
    for arm, reference in cfg["calibration"].items():
        gap = results["tasks"]["piqa"][arm]["percent"] - reference
        results["calibration"][arm] = dict(reference=reference, delta_pp=gap, audit_required=abs(gap)>1)
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / "metrics.json", results)
    save(folder / "protocols.json", protocols)
    files = ["metrics.json", "protocols.json", "accuracy.png", "paired-differences.png"]
    for task, arms in corpus.items():
        name = f"{task}-predictions.jsonl.gz"
        with (
            (folder / name).open("wb") as stream,
            gzip.GzipFile(fileobj=stream, mode="wb", mtime=0, filename="") as zipped,
        ):
            for arm, rows in arms.items():
                for row in rows:
                    zipped.write(
                        (
                            json.dumps(
                                dict(model=arm, **row),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode()
                    )
        files.append(name)
    figures(folder, results)
    write_summaries(folder, results)
    save(
        folder / "manifest.json",
        dict(
            files={f: sha(folder / f) for f in files},
            assets={n: h for n, (u, h) in assets(cfg).items()},
        ),
    )
    return results


def write_summaries(folder, results):
    for filename in ("README.md", "REPORT.md"):

        lines = [
            "# Public evaluation of six Qwen checkpoints",
            "[Evaluation results](metrics.json)",
            "",
            "The primary comparison uses two adapters trained from SFT B on the same 166 prompts. Continued SFT learned the chosen answers and was selected at step 11. Upgraded DPO also trained on the model's rejected answers and was selected at step 6. The frozen SFT B and previous DPO checkpoints show how earlier stages performed. The two official Qwen checkpoints provide external baselines.",
        ]
        tasks = results["tasks"]
        primary = {
            task: results["comparisons"][task]["sft-42"]["primary"] for task in tasks
        }
        lines.extend(
            [
                "",
                "Upgraded DPO versus continued SFT changed PIQA, HellaSwag and IFEval strict prompt accuracy by "
                + ", ".join(f"{p['delta_pp']:+.2f}" for p in primary.values())
                + " percentage points. Each model was selected before running these public tasks.",
                "",
                "| Checkpoint | PIQA | HellaSwag | IFEval strict prompt |",
                "|---|---:|---:|---:|",
            ]
        )
        for arm, label in NAMES.items():
            cells = []
            for task in tasks:
                m = tasks[task].get(arm)
                cells.append(
                    f"{m['correct']}/{m['total']} ({m['percent']:.2f}%)" if m else "—"
                )
            lines.append("| " + " | ".join([label, *cells]) + " |")
        lines.extend(
            [
                "",
                "![Accuracy by task](accuracy.png)",
                "",
                "The PIQA references are 69.9% for Base and 67.2% for Instruct. The largest absolute difference from these references was {:.2f} percentage points. The SmolLM2 card lists a prompt/instruction average for IFEval. We report all four upstream scores separately."
                .format(max(abs(x["delta_pp"]) for x in results["calibration"].values()))
                + " [SmolLM2 model card](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct#evaluation)",
                "",
                "## Paired changes",
                "",
                "| Task | Upgraded DPO compared with | Gained | Lost | Difference (pp) | 95% interval (pp) |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for task, comparisons in results["comparisons"].items():
            for control, values in comparisons.items():
                p = values["primary"]
                lo, hi = p["ci95"]
                lines.append(
                    f"| {task} | {NAMES[control]} | {p['gained']} | {p['lost']} | {p['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |"
                )
        inconclusive = [
            t for t, p in primary.items() if p["ci95"][0] <= 0 <= p["ci95"][1]
        ]
        lines.extend(["", "![DPO minus continued SFT](paired-differences.png)", ""])
        if inconclusive:
            lines.append(
                "The paired intervals include zero on "
                + ", ".join(inconclusive)
                + ". These comparisons give no clear evidence of a DPO gain."
            )
        lines.extend(
            [
                "Intervals resample complete prompts 10,000 times with seed 42. They measure uncertainty over questions for the selected checkpoints. Training used a single seed.",
                "",
                "## IFEval",
                "",
                "| Checkpoint | Strict prompt | Strict instruction | Loose prompt | Loose instruction | Length-limited / 541 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for arm, m in tasks["ifeval"].items():
            lines.append(
                "| "
                + " | ".join(
                    [
                        NAMES[arm],
                        *[f"{m[k]['percent']:.2f}%" for k in IF_KEYS],
                        str(m["truncated"]),
                    ]
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "## Changed answers",
                "",
                "For each direction we show the first changed question in the frozen order. Choice numbers below start at one. Full answers and candidate scores are in the prediction files.",
                "",
                "| Task | DPO change against continued SFT | ID | Observed change |",
                "|---|---|---|---|",
            ]
        )
        for task, cases in results["cases"].items():
            for direction, case in cases.items():
                label = direction
                if case is None:
                    lines.append(f"| {task} | {label} | — | 0 |")
                    continue
                a, b = case["continued_sft"], case["upgraded_dpo"]
                if task == "ifeval":

                    def failures(x):
                        return ", ".join(
                            k
                            for k, ok in zip(
                                case["instructions"],
                                x["scores"]["inst_level_strict_acc"],
                            )
                            if not ok
                        ) or "all pass"

                    change = (
                        "Failed checks (SFT → DPO) "
                        + failures(a)
                        + " → "
                        + failures(b)
                    )
                else:
                    change = (
                        "Choice "
                        + f"{a['pred'] + 1} → {b['pred'] + 1}"
                        + ", gold "
                        + str(a["gold"] + 1)
                    )
                lines.append(f"| {task} | {label} | {case['id']} | {change} |")
        lines.extend(
            [
                "",
                "## Runtime and reproduction",
                "",
                "| Task | Model/question records | Synchronized batch time (min) | Peak allocated GPU memory (MiB) |",
                "|---|---:|---:|---:|",
            ]
        )
        for task, arms in tasks.items():
            lines.append(
                f"| {task} | {sum(r['total'] for r in arms.values()):,} | {sum(r['seconds'] for r in arms.values()) / 60:.1f} | {max(r['peak_mib'] for r in arms.values()):.0f} |"
            )
        lines.extend(
            [
                "",
                "Single RTX 3060 Laptop GPU (6 GB), Transformers BF16. Eight-question checks used batch one, formal runs used batch eight. Greedy outputs can change with batching. The first-eight agreement and per-checkpoint costs are recorded in metrics.json.",
                "",
                "HellaSwag validation uses the author's frozen Parquet release. The original loader's GitHub data URL is unavailable. Raw-file equivalence to the old JSONL could not be verified.",
                "",
                "[Reproduction commands](reproduce.md) · [Metrics and cases](metrics.json) · [Protocol fingerprints](protocols.json) · [File hashes](manifest.json)",
                "",
                "[PIQA predictions](piqa-predictions.jsonl.gz) · [HellaSwag predictions](hellaswag-predictions.jsonl.gz) · [IFEval responses and checks](ifeval-predictions.jsonl.gz) · [Paired reasoning experiment](../paired-dpo-results/README.md)",
            ]
        )
        (folder / filename).write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )


def verify(cfg, folder, rescore=False):
    """Rebuild every published score from the matching public records and question IDs."""
    manifest = read(folder / "manifest.json")
    if any(
        sha(folder / name) != checksum for name, checksum in manifest["files"].items()
    ):
        raise ValueError("Published artifact checksum mismatch")
    report = read(folder / "metrics.json")
    protocols = read(folder / "protocols.json")
    if rescore:
        from langdetect import DetectorFactory

        DetectorFactory.seed = cfg["seed"]
        asset_dir = Path(cfg["assets"]).resolve()
        os.environ.setdefault("NLTK_DATA", str(asset_dir / "nltk_data"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(asset_dir / "hub"))
        from .smollm2_eval import score_instruction
    total = 0
    for task, settings in cfg["tasks"].items():
        src = read(Path(cfg["root"]) / f"{task}.json")
        expected = set(NAMES) - ({"base"} if task == "ifeval" else set())
        arms = {n: [] for n in expected}
        with gzip.open(
            folder / f"{task}-predictions.jsonl.gz", "rt", encoding="utf-8"
        ) as stream:
            for line in stream:
                record = json.loads(line)
                arms[record.pop("model")].append(record)
        if set(report["tasks"][task]) != expected:
            raise ValueError("Unexpected set of models")
        for model, rows in arms.items():
            meta = protocols[task][model].copy()
            key = meta.pop("key")
            if (
                digest(meta) != key
                or meta["data"] != digest(src)
                or len(rows) != settings["n"]
            ):
                raise ValueError("Source data or protocol mismatch")
            random.seed(cfg["seed"])
            for r, s in zip(rows, src, strict=True):
                verify_row(task, r, s)
                if (
                    task == "ifeval"
                    and rescore
                    and score_instruction(s, r["response"]) != r["scores"]
                ):
                    raise ValueError(f"IFEval score mismatch: {model}, {r['id']}")
            stats = report["tasks"][task][model]
            if (
                stats["correct"] != sum(r["correct"] for r in rows)
                or stats["total"] != len(rows)
                or stats["percent"] != 100 * stats["correct"] / len(rows)
            ):
                raise ValueError("Reported aggregate is inconsistent")
            if task == "ifeval":
                for k in IF_KEYS:
                    a, n = units(rows, k).sum(axis=0)
                    if stats[k] != dict(passed=a, total=n, percent=100 * a / n):
                        raise ValueError("IFEval aggregation mismatch")
            total += len(rows)
        for model in CONTROLS:
            primary = "prompt_level_strict_acc" if task == "ifeval" else "accuracy"
            for key, value in report["comparisons"][task][model].items():
                if (
                    paired(
                        arms["dpo-42"],
                        arms[model],
                        primary if key == "primary" else key,
                        cfg["seed"],
                    )
                    != value
                ):
                    raise ValueError("Paired statistics mismatch")
    print(f"Verified {total:,} model/question records.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "render"])
    parser.add_argument(
        "--config", type=Path, default=Path("configs/smollm2-public.json")
    )
    parser.add_argument("--artifacts", type=Path, default=Path("docs/smollm2-results"))
    parser.add_argument("--output", type=Path, default=Path("outputs/smollm2-figures"))
    parser.add_argument("--rescore-ifeval", action="store_true")
    args = parser.parse_args()
    if args.command == "verify":
        verify(read(args.config), args.artifacts, args.rescore_ifeval)
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        figures(args.output, read(args.artifacts / "metrics.json"))
