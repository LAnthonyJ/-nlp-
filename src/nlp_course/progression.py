"""Run the preregistered one-SFT, one-DPO experiment, resuming committed batches."""

import argparse
import gc
import os
import random
import time
from pathlib import Path

import torch
from .capability import read, sha
from .capability_data import digest
from .progression_data import TASKS, prepare
from .progression_engine import (
    evaluate,
    generate,
    load_model,
    logps,
    mine,
    prompt_for,
    train,
)
from .progression_stats import compare, scores
from .smollm2_data import freeze


def take(rows, n, extra=None):
    return [
        r
        for task in TASKS
        for r in [x for x in rows if x["task"] == task][
            : n * (2 if task == extra else 1)
        ]
    ]


def smoke(cfg, rows, folder):
    """Exercise completion masking, optimizer, adapter reload and deterministic generation."""
    result = folder / "verified.json"
    if result.exists() and read(result)["passed"]:
        return
    model, tok = load_model(cfg, trainable=True)
    selected = sorted(
        (r for r in rows if r["task"] == "gsm8k"), key=lambda r: len(r["prompt"])
    )[:16]
    # Training-only abbreviated answers keep this wiring test inexpensive.
    pairs = [(prompt_for(r), " " + r["gold"]) for r in selected]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    for step in range(160):
        batch = pairs[(step % 4) * 4 : (step % 4 + 1) * 4]
        model.train()
        optimizer.zero_grad()
        sums, n = logps(model, tok, batch)
        loss = -(sums / n).mean()
        loss.backward()
        optimizer.step()
    model.save_pretrained(folder)
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    model, tok = load_model(cfg, folder)
    outputs = generate(model, tok, selected, 16)
    correct = sum(
        g["response"].strip() == r["gold"]
        for g, r in zip(outputs, selected, strict=True)
    )
    if correct < 14:
        raise RuntimeError(f"16-prompt reload test: {correct}/16")
    freeze(result, dict(passed=True, correct=correct, n=16))


def change(first, second, point):
    if not first["passed"]:
        changes = first["tasks"]
        if all(v["pp"] <= 0 for v in changes.values()):
            return "sft_lr"
        return "more_" + min(changes, key=lambda k: changes[k]["pp"])
    if not second["passed"] and point.get("chosen_delta", 0) < 0:
        return "dpo_lr"
    return "hard_pairs"


def reuse_fixed_inputs(previous, current):
    """A learning-rate-only DPO retry uses exactly the previous frozen inputs."""
    outcome = read(previous / "outcome.json")
    old, new = (read(folder / "design.json") for folder in (previous, current))
    if (
        outcome["action"] not in ("hard_pairs", "dpo_lr")
        or old["dpo_lr"] == new["dpo_lr"]
    ):
        return
    if new["sft_parent"] != outcome["sft"] or any(
        old[k] != new[k] for k in ("hard", "dev", "extra", "sft_lr")
    ):
        return
    weights = Path(outcome["sft"]) / "adapter_model.safetensors"
    if sha(weights) != read(weights.parent.parent / "selected.json")["weights"]:
        raise ValueError("Cannot reuse predictions from changed SFT weights")
    for name in ("base-dev", "selected-sft-dev", "mining"):
        for source in (previous / name).rglob("*.json"):
            destination = current / source.relative_to(previous)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, destination)
            except FileExistsError:
                if sha(source) != sha(destination):
                    raise ValueError(f"Conflicting cache at {destination}")
    freeze(
        current / "reuse.json",
        dict(
            source=previous.name,
            weights=sha(weights),
            preferences=sha(previous / "mining/pairs.json"),
        ),
    )


def run(cfg):
    from langdetect import DetectorFactory

    DetectorFactory.seed = cfg["seed"]
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("This experiment requires a BF16 CUDA GPU")
    root = Path(cfg["root"])
    manifest = read(root / "manifest.json")
    if manifest["config"] != digest(cfg):
        raise ValueError("Configuration changed after freeze")
    banks = {p: read(root / f"bank/{p}.json") for p in ("sft", "dev", "mine")}
    if any(digest(v) != manifest["banks"][k] for k, v in banks.items()):
        raise ValueError("Bank changed")
    base_dir = os.environ.get("PROGRESSION_BASE")
    if (
        base_dir
        and sha(Path(base_dir) / "model.safetensors")
        != read("docs/repair-results/protocols.json")["base"]["weights"]["base"]
    ):
        raise ValueError("Base checksum mismatch")
    freeze(
        root / "environment.json",
        dict(
            model=cfg["model"],
            revision=cfg["revision"],
            gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            bf16=True,
            seed=cfg["seed"],
        ),
    )
    smoke(cfg, banks["sft"], root / "smoke")
    sft_lr = cfg["sft_lr"]
    dpo_lr = cfg["dpo_lr"]
    extra = None
    hard = False
    dev_n = cfg["dev_per_task"]
    sft = None
    opened = 0
    for attempt in range(1, cfg["rounds"] + 1):
        folder = root / f"round-{attempt}"
        folder.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        dev = take(banks["dev"], dev_n)
        freeze(
            folder / "design.json",
            dict(
                sft_lr=sft_lr,
                dpo_lr=dpo_lr,
                extra=extra,
                hard=hard,
                dev=dev_n,
                sft_parent=str(sft) if sft else cfg["revision"],
            ),
        )
        if (folder / "outcome.json").exists():
            result = read(folder / "outcome.json")
            sft = Path(result["sft"])
        else:
            if attempt > 1:
                reuse_fixed_inputs(root / f"round-{attempt - 1}", folder)
            model, tok = load_model(cfg)
            print(
                f"Round {attempt}: measuring Base on {len(dev)} development prompts",
                flush=True,
            )
            base_rows = evaluate(cfg, model, tok, dev, folder / "base-dev")
            del model
            torch.cuda.empty_cache()
            if sft is None:
                print("Training SFT", flush=True)
                sft = train(
                    cfg,
                    "sft",
                    None,
                    take(banks["sft"], cfg["train_per_task"], extra),
                    dev,
                    folder / "sft",
                    sft_lr,
                )
            model, tok = load_model(cfg, sft)
            sft_rows = evaluate(cfg, model, tok, dev, folder / "selected-sft-dev")
            first = compare(base_rows, sft_rows, 2)
            print("Mining SFT preferences", flush=True)
            pairs = mine(cfg, model, tok, banks["mine"], folder / "mining", hard)
            support = {task: sum(r["task"] == task for r in pairs) for task in TASKS}
            del model
            torch.cuda.empty_cache()
            result = dict(
                sft=str(sft), pairs=support, dev_sft=first, passed=False, opened=False
            )
            if min(support.values()) < cfg["min_pairs"]:
                # Retry generation with greater diversity from this SFT, at the same prompt cap.
                result.update(
                    reason="insufficient verified preference pairs", action="sampling"
                )
            else:
                print("Training DPO", flush=True)
                dpo = train(cfg, "dpo", sft, pairs, dev, folder / "dpo", dpo_lr)
                point = read(dpo / "summary.json")
                model, tok = load_model(cfg, dpo)
                dpo_rows = evaluate(cfg, model, tok, dev, folder / "selected-dpo-dev")
                second = compare(sft_rows, dpo_rows, 1)
                del model
                torch.cuda.empty_cache()
                result.update(
                    dpo=str(dpo),
                    dev_dpo=second,
                    scores={
                        k: scores(v)
                        for k, v in zip(
                            ("base", "sft", "dpo"),
                            (base_rows, sft_rows, dpo_rows),
                            strict=True,
                        )
                    },
                    action=change(first, second, point),
                )
                if first["passed"] and second["passed"]:
                    panel_id = opened + 1
                    lock = dict(
                        attempt=attempt,
                        panel=panel_id,
                        base=cfg["revision"],
                        sft=sha(sft / "adapter_model.safetensors"),
                        dpo=sha(dpo / "adapter_model.safetensors"),
                    )
                    freeze(root / f"opened/{panel_id}.json", lock)
                    panel = read(root / f"panels/{panel_id}.json")
                    if digest(panel) != manifest["panels"][panel_id - 1]:
                        raise ValueError("Acceptance panel changed")
                    preds = []
                    for name, adapter in [("base", None), ("sft", sft), ("dpo", dpo)]:
                        model, tok = load_model(cfg, adapter)
                        preds.append(
                            evaluate(cfg, model, tok, panel, folder / f"formal-{name}")
                        )
                        del model
                        torch.cuda.empty_cache()
                    a, b = (
                        compare(preds[0], preds[1], 2, True),
                        compare(preds[1], preds[2], 1, True),
                    )
                    result.update(
                        formal_sft=a,
                        formal_dpo=b,
                        passed=a["passed"] and b["passed"],
                        opened=True,
                        panel=panel_id,
                        action="expand_dev",
                        formal_scores=dict(
                            zip(("base", "sft", "dpo"), map(scores, preds), strict=True)
                        ),
                    )
            result["elapsed_seconds"] = time.monotonic() - start
            freeze(folder / "outcome.json", result)
        print(result, flush=True)
        opened += int(result["opened"])
        if result["passed"] or attempt == cfg["rounds"]:
            freeze(
                root / "finished.json",
                dict(attempts=attempt, passed=result["passed"], opened=opened),
            )
            return
        action = result["action"]
        if action == "sampling":
            cfg = dict(
                cfg,
                mining_temperature=min(1.6, cfg.get("mining_temperature", 0.85) + 0.35),
            )
        elif action == "more_" + str(extra) or action == "sft_lr":
            sft_lr /= 2
            sft = None
        elif action.startswith("more_"):
            extra = action[5:]
            sft = None
        elif action == "expand_dev":
            dev_n = 2 * cfg["dev_per_task"]
            sft = None
        elif action == "dpo_lr" or (action == "hard_pairs" and hard):
            dpo_lr /= 2
        else:
            hard = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "public", "report"))
    parser.add_argument("--config", default="configs/progression.json")
    args = parser.parse_args()
    cfg = read(args.config)
    dispatch(args.command, cfg)


def dispatch(command, cfg):
    if command == "prepare":
        prepare(cfg)
    elif command == "public":
        from .progression_report import public

        public(cfg)
    elif command == "report":
        from .progression_report import report

        report(Path(cfg["root"]))
    else:
        # Kernel releases this single-run lock if a cloud process dies.
        import fcntl

        with (Path(cfg["root"]) / "run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(cfg)


if __name__ == "__main__":
    main()
