"""Deterministic, split-specific paired tasks and scoring protocol."""

import itertools
import random
import re
from collections import defaultdict

KINDS = ("judgment", "ordering", "counting")
QUOTAS = dict(zip(KINDS, (96, 64, 64)))
SPLITS = ("train", "dev", "holdout")
PROTOCOL = "paired-exact-v1"


def unique_order(names, edges):
    solutions = [
        order
        for order in itertools.permutations(names)
        if all(order.index(a) < order.index(b) for a, b in edges)
    ]
    if len(solutions) != 1:
        raise ValueError("Expected exactly one valid order")
    return ",".join(solutions[0])


def variants(kind, split, i, rng):
    level = SPLITS.index(split)
    if kind == "judgment":
        names = rng.sample(["oak", "birch", "pine", "elm", "maple", "ash", "cedar"], 5)
        a, b, c, d, e = names
        # One, two, or three implications; the alternative starting fact is disconnected.
        chain = names[: level + 2]
        rules = [f"Every {x} is a {y}." for x, y in zip(chain, chain[1:])]
        rng.shuffle(rules)
        for fact in (a, e):
            edges = dict(zip(chain, chain[1:]))
            reached = fact
            while reached in edges:
                reached = edges[reached]
            yield (
                f"Specimen {i} is a {fact}. "
                + " ".join(rules)
                + f" Does it necessarily follow that specimen {i} is a {chain[-1]}? Answer yes or no only.",
                "yes" if reached == chain[-1] else "no",
            )
    elif kind == "ordering":
        names = rng.sample([chr(65 + j) for j in range(26)], 3 + level)
        a, b, c, *tail = names
        common = [(a, c), (b, c), *zip([c] + tail, tail)]
        rng.shuffle(common)
        pos = rng.randrange(len(common) + 1)
        for edge in [(a, b), (b, a)]:
            edges = common[:pos] + [edge] + common[pos:]
            yield (
                " ".join(f"{x} precedes {y}." for x, y in edges)
                + " Give the complete order using comma-separated letters only.",
                unique_order(names, edges),
            )
    else:
        x, y, z = rng.sample(range(4, 40), 3)
        change = rng.randint(1, min(x, y) - 1)
        if split == "train":
            facts = (
                f"A bin has {x} red and {y} blue beads. Take away {change} red beads."
            )
            before, other = x - change, y
        elif split == "dev":
            facts = f"A shop has {x} red and {y} blue beads. It receives {change} red beads."
            before, other = x + change, y
        else:
            facts = f"A tray holds {x} red, {y} blue and {z} white beads. Move {change} red beads onto another tray."
            before, other = x - change, y + z
        for scope, answer in [
            ("red beads", before),
            ("beads of all colors", before + other),
        ]:
            yield (
                f"{facts} How many {scope} are there now in the original container? Return only an integer.",
                str(answer),
            )


def build_suite():
    suite = {}
    seen = set()
    for split in SPLITS:
        rng = random.Random(4100 + SPLITS.index(split))
        tasks = []
        for kind in KINDS:
            n = 2 * QUOTAS[kind] if split == "train" else (10 if split == "dev" else 20)
            group = 0
            while group < n:
                rows = list(variants(kind, split, group, rng))
                if any(p in seen for p, _ in rows):
                    continue
                if len({a for _, a in rows}) != 2:
                    raise ValueError("Condition failed to change reference")
                rng.shuffle(rows)
                for side, (prompt, answer) in enumerate(rows):
                    seen.add(prompt)
                    key = f"{split}-{kind}-{group}"
                    tasks.append(
                        dict(
                            id=f"{key}-{side}",
                            group=key,
                            kind=kind,
                            prompt=prompt,
                            answer=answer,
                        )
                    )
                group += 1
        suite[split] = tasks
    return suite


def assess(task, response, truncated):
    # Only a single complete parseable response can supply an unambiguous negative.
    raw = response.strip()
    pattern = {
        "judgment": r"yes|no",
        "ordering": r"[A-Z](?:\s*,\s*[A-Z])+",
        "counting": r"-?[0-9]+",
    }[task["kind"]]
    valid = not truncated and bool(re.fullmatch(pattern, raw, re.I))
    normalized = raw.replace(" ", "").lower()
    return dict(format_ok=valid, correct=valid and normalized == task["answer"].lower())


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(row)
    if not groups or any(
        len(pair) != 2 or len({r["id"] for r in pair}) != 2 for pair in groups.values()
    ):
        raise ValueError("Incomplete or duplicated task pair")
    wins = {group: all(r["correct"] for r in pair) for group, pair in groups.items()}
    return dict(
        groups=wins,
        both=sum(wins.values()),
        n=len(wins),
        single=sum(r["correct"] for r in rows),
        format_ok=sum(r["format_ok"] for r in rows),
        by_kind={
            k: sum(wins[g] for g, p in groups.items() if p[0]["kind"] == k)
            for k in KINDS
        },
    )
