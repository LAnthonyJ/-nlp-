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
    arms = {}
    for name in ("base", "sft", "dpo"):
        items = []
        for p in sorted((root / "public" / name).glob("*/*.json")):
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
        if {r["id"] for r in items} != set(questions) or scores(items) != full[
            "scores"
        ][name]:
            raise ValueError("Incomplete public evaluation")
        arms[name] = items
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
        "# Qwen2.5-0.5B 单次 SFT 与单次 DPO 的公开评测",
        "",
        f"共尝试 {final['attempts']} 轮，正式验收{'通过' if final['passed'] else '未通过'}。下表来自锁定最终权重后的全量描述性评测。",
        "",
        "| 阶段 | PIQA | HellaSwag | IFEval 严格提示级 | GSM8K | 等权均分 |",
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
            "## 两段公开能力变化",
            "",
            "| 阶段 | 综合分差（百分点） | 配对 95% 区间 | 回退最大的任务及变化（百分点） |",
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
    lines.extend(
        [
            "",
            "## 迭代记录",
            "",
            "| 轮次 | SFT 开发增益 | DPO 开发增益 | 通过开发门槛后开封的集合 | 单卡运行分钟 |",
            "|---|---:|---:|---|---:|",
        ]
    )
    for i, r in enumerate(history, 1):
        d = f"{r['dev_dpo']['pp']:+.2f}" if "dev_dpo" in r else "偏好对不足"
        lines.append(
            f"| {i} | {r['dev_sft']['pp']:+.2f} | {d} | {r.get('panel', '—')} | {r['elapsed_seconds'] / 60:.1f} |"
        )
    lines.extend(
        [
            "",
            "开发增益以百分点计。单卡运行时间含本轮开发评估与偏好采集。各次重训的配置、逐项错转对与对转错题数、正负回答的概率变化见指标文件。",
            "",
            "[训练与复现说明](../progression.md)",
        ]
    )
    lines.extend(
        [
            "",
            "![综合分与配对区间](progression.png)",
            "",
            "![各项得分](tasks.png)",
            "",
            "完整记录包括未达标的轮次、正式验收状态、封存题哈希、逐题输出与改善和退化案例。公开题存在历史暴露，结论限定在本次单种子权重和冻结协议。",
            "",
            "[指标与验收](metrics.json) · [逐题结果](predictions.jsonl.gz) · [变化案例](cases.json)",
        ]
    )
    (dest / "README.zh-CN.md").write_text("\n".join(lines) + "\n", encoding="utf8")
