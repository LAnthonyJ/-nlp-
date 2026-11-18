import gzip
import json
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from nlp_course.paired_tasks import assess, build_suite, summarize, unique_order
from nlp_course.paired_experiment import (
    difference,
    paired_logps,
    select_pair,
    validate_pairs,
)
from nlp_course.capability import read, sha
from nlp_course.capability_data import digest
from nlp_course.training import sequence_logps


class PairedTests(unittest.TestCase):
    def test_freeze_is_deterministic_and_siblings_share_a_split(self):
        suite = build_suite()
        self.assertEqual(suite, build_suite())
        seen = set()
        seen_templates = set()
        for split, tasks in suite.items():
            self.assertTrue(seen.isdisjoint(t["prompt"] for t in tasks))
            seen.update(t["prompt"] for t in tasks)
            templates = set()
            for row in tasks:
                prompt = row["prompt"]
                if row["kind"] == "judgment":
                    structure = prompt.count("Every ")
                elif row["kind"] == "ordering":
                    structure = prompt.count(" precedes ")
                else:
                    structure = re.sub(r"[0-9]+", "#", prompt.split(" How many ")[0])
                templates.add((row["kind"], structure))
            self.assertTrue(seen_templates.isdisjoint(templates))
            seen_templates.update(templates)
            for a, b in zip(tasks[::2], tasks[1::2]):
                self.assertEqual(a["group"], b["group"])
                self.assertTrue(a["group"].startswith(split + "-"))
                self.assertNotEqual(a["answer"], b["answer"])
                if a["kind"] == "ordering":
                    self.assertEqual(
                        len(a["answer"].split(",")),
                        3 + ["train", "dev", "holdout"].index(split),
                    )
        self.assertEqual(len(suite["dev"]), 60)
        self.assertEqual(len(suite["holdout"]), 120)

    def test_order_must_be_unique(self):
        self.assertEqual(unique_order("ABC", [("C", "A"), ("A", "B")]), "C,A,B")
        with self.assertRaises(ValueError):
            unique_order("ABC", [("A", "B")])

    def test_negative_must_be_observed_unambiguous_and_complete(self):
        task = dict(kind="judgment", answer="no")

        def result(s, truncated=False):
            return dict(response=s, truncated=truncated)

        self.assertIsNone(
            select_pair(task, [result("yes or no"), result("yes", True), result("no")])
        )
        pair = select_pair(task, [result("Yes"), result("NO")])
        self.assertEqual(
            (pair["chosen"], pair["rejected"], pair["chosen_source"]),
            ("NO", "Yes", "model"),
        )
        self.assertEqual(select_pair(task, [result("Yes")])["chosen_source"], "solver")

    def test_pair_requires_both_answers(self):
        rows = []
        for i, answer in enumerate(("no", "yes")):
            task = dict(kind="judgment", id=str(i), group="one", answer=answer)
            rows.append(dict(**task, **assess(task, "yes", False)))
        result = summarize(rows)
        self.assertEqual((result["single"], result["both"]), (1, 0))
        with self.assertRaises(ValueError):
            summarize(rows[:1])
        self.assertEqual(
            difference({"one": True}, {"one": False})["improved_ids"], ["one"]
        )

    def test_cached_reference_does_not_follow_policy_update(self):
        class TinyLM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.zeros(1, 3, 4))

            def forward(self, **kwargs):
                return SimpleNamespace(logits=self.logits)

        model = TinyLM()
        ex = dict(input_ids=[0, 1, 2], labels=[-100, 1, 2])
        cache = paired_logps(model, [(ex, ex)], "cpu")
        reference = cache[0][0]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        (-sequence_logps(model, ex, "cpu")).backward()
        optimizer.step()
        self.assertGreater(sequence_logps(model, ex, "cpu").item(), reference)
        self.assertEqual(cache, [(reference, reference)])

    def test_published_pairs_match_the_frozen_model_candidates(self):
        folder = Path(__file__).resolve().parents[1] / "docs/paired-dpo-results"
        pairs = read(folder / "pairs.json")
        suite = build_suite()
        validate_pairs(pairs, suite["train"])
        with gzip.open(folder / "mining.json.gz", "rt", encoding="utf-8") as stream:
            audit = {row["id"]: row["candidates"] for row in json.load(stream)}
        for pair in pairs:
            candidates = audit[pair["id"]]
            self.assertIn(
                pair["rejected"],
                [
                    r["response"].strip()
                    for r in candidates
                    if r["format_ok"] and not r["correct"]
                ],
            )
            if pair["chosen_source"] == "model":
                self.assertIn(
                    pair["chosen"],
                    [r["response"].strip() for r in candidates if r["correct"]],
                )
            else:
                self.assertEqual(pair["chosen"], pair["answer"])

    def test_published_counts_follow_from_raw_generations(self):
        folder = Path(__file__).resolve().parents[1] / "docs/paired-dpo-results"
        suite = build_suite()
        report = read(folder / "metrics.json")
        self.assertEqual(digest(suite), report["protocol"]["suite_hash"])
        for filename, expected in read(folder / "manifest.json").items():
            self.assertEqual(sha(folder / filename), expected)
        for name, rows in read(folder / "generations.json").items():
            self.assertEqual(len(rows), len(suite["holdout"]))
            for task, row in zip(suite["holdout"], rows):
                self.assertEqual(task, {key: row[key] for key in task})
                for key, value in assess(
                    task, row["response"], row["truncated"]
                ).items():
                    self.assertEqual(row[key], value)
            self.assertEqual(summarize(rows), report["final"][name])
