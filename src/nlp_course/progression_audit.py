"""Audit the actual supervised token span against archived SFT and preference labels."""

import math
import os
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from .capability import read, sha
from .capability_data import digest
from .progression import take
from .progression_engine import completion_tokens, grade, prompt_for, split_tokens
from .smollm2_data import freeze
from .training import preference_loss


def audit(cfg):
    root = Path(cfg["root"]) / "audit"
    prior = Path(cfg["prior_inputs"])
    archive = Path(cfg["archive"])
    tok = AutoTokenizer.from_pretrained(
        os.getenv("PROGRESSION_BASE") or cfg["model"],
        revision=cfg["revision"],
        local_files_only=True,
    )
    tok.completion_v2 = True
    banks = {name: read(prior / "bank" / (name + ".json")) for name in ("sft", "mine")}
    manifest = read(archive / "manifest.json")
    for name, rows in banks.items():
        if digest(rows) != manifest["banks"][name]:
            raise ValueError("Historical bank mismatch")
    pairs = read(archive / "round-2/mining/pairs.json")
    source = {r["id"]: r for r in banks["mine"]}
    ordered = list(pairs)
    import random

    random.Random(42).shuffle(ordered)
    parent = sha(archive / "round-1/sft/step-32/adapter_model.safetensors")
    if (
        parent != "df91a12d3997db305d9d1fc6c89c36d4c198fc535fed65538a17202901d0dc64"
        or digest([parent, ordered])
        != read(archive / "round-2/dpo/reference.json")["key"]
    ):
        raise ValueError("Historical parent or preference order mismatch")
    records = []
    counts = Counter()

    def check(row, prompt, side, value):
        prefix, suffix = completion_tokens(tok, prompt, value, row["task"])
        opts = {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        actual = tok.decode(suffix[:-1], **opts)
        old = tok.decode(split_tokens(tok, prompt, value)[1], **opts)
        correct = grade(row, actual)
        old_correct = grade(row, old)
        if (
            actual != value
            or correct != (side != "rejected")
            or suffix[-1] != tok.eos_token_id
            or prefix != tok(prompt).input_ids
        ):
            raise ValueError(f"Supervision mismatch: {row['id']} {side}")
        counts[side] += 1
        if row["task"] == "ifeval" and correct and not old_correct:
            counts["restored_" + side] += 1
        records.append(
            {
                "id": row["id"],
                "side": side,
                "original": value,
                "decoded": actual,
                "legacy": old,
                "old_correct": old_correct,
                "correct": correct,
            }
        )

    for row in take(banks["sft"], 1024):
        if "choices" not in row:
            check(row, prompt_for(row), "answer", row["answer"])
    for pair in pairs:
        row = source[pair["id"]]
        if "choices" in row:
            for answer in row["choices"]:
                if completion_tokens(
                    tok, row["prompt"], answer, row["task"], False
                ) != split_tokens(tok, row["prompt"], answer):
                    raise ValueError("Choice scoring changed")
            counts["unchanged_choice_prompts"] += 1
        else:
            for side in ("chosen", "rejected"):
                check(row, pair["prompt"], side, pair[side])
    if [
        counts[k]
        for k in (
            "chosen",
            "rejected",
            "restored_chosen",
            "restored_answer",
            "unchanged_choice_prompts",
        )
    ] != [512, 512, 136, 643, 512]:
        raise ValueError(f"Unexpected archive coverage: {counts}")
    a, b = [torch.tensor([v], requires_grad=True) for v in (-2.0, -3.0)]
    ra, rb = a.detach().clone(), b.detach().clone()
    loss = preference_loss(a, b, ra, rb, 0.1)
    loss.backward()
    with torch.no_grad():
        a.sub_(a.grad)
        b.sub_(b.grad)
    if not (
        math.isclose(loss.item(), math.log(2), abs_tol=1e-6)
        and a > ra
        and b < rb
        and (ra.item(), rb.item()) == (-2.0, -3.0)
    ):
        raise ValueError("DPO gradient or reference mismatch")
    freeze(root / "rows.json", records)
    summary = {
        "counts": dict(counts),
        "protocol": cfg["protocol"],
        "tokenizer": digest(tok.backend_tokenizer.to_str()),
        "parent": parent,
        "pairs": digest(pairs),
        "rows": digest(records),
        "passed": True,
    }
    freeze(root / "verified.json", summary)
    print(summary, flush=True)
