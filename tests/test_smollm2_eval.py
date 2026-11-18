import gzip
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nlp_course.smollm2_data import freeze
from nlp_course.smollm2_eval import commit_batch, connect, ordered, template_prompt
from nlp_course.training import prompt_text
from nlp_course.capability import read, sha


class ProtocolTests(unittest.TestCase):
    def test_published_matrix_recomputes_without_gpu_or_datasets(self):
        folder = Path(__file__).resolve().parents[1] / "docs/smollm2-results"
        manifest = read(folder / "manifest.json")
        for filename, expected in manifest["files"].items():
            self.assertEqual(sha(folder / filename), expected)
        tasks = read(folder / "metrics.json")["tasks"]
        self.assertEqual(sum(map(len, tasks.values())), 17)
        for task, arms in tasks.items():
            ids = {name: [] for name in arms}
            passed = dict.fromkeys(arms, 0)
            with gzip.open(
                folder / f"{task}-predictions.jsonl.gz", "rt", encoding="utf-8"
            ) as stream:
                for line in stream:
                    r = json.loads(line)
                    if task == "ifeval":
                        correct = all(r["scores"]["inst_level_strict_acc"])
                    else:
                        scores = r["normalized"]
                        correct = (
                            max(range(len(scores)), key=scores.__getitem__) == r["gold"]
                        )
                    ids[r["model"]].append(r["id"])
                    passed[r["model"]] += correct
                    self.assertEqual(correct, r["correct"])
            reference = next(iter(ids.values()))
            self.assertEqual(len(set(reference)), len(reference))
            for model, expected in arms.items():
                self.assertEqual(ids[model], reference)
                self.assertEqual(
                    (passed[model], len(reference)),
                    (expected["correct"], expected["total"]),
                )

    def test_resume_preserves_order_and_rejects_mismatched_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            db = connect(Path(directory) / "results.sqlite")
            rows = [{"id": "first"}, {"id": "second"}]
            try:
                commit_batch(db, "experiment", rows[1:], 1.0, 99)
                self.assertEqual(
                    ordered(db, "experiment", rows, complete=False), rows[1:]
                )
                with self.assertRaises(ValueError):
                    ordered(db, "experiment", rows)
                with self.assertRaises(sqlite3.IntegrityError):
                    commit_batch(db, "experiment", rows, 2.0, 100)
                self.assertEqual(
                    db.execute("SELECT count(*) FROM costs").fetchone()[0], 1
                )
                commit_batch(db, "experiment", rows[:1], 1.0, 99)
                self.assertEqual(ordered(db, "experiment", rows), rows)
                with self.assertRaises(ValueError):
                    ordered(db, "experiment", rows[:1])
            finally:
                db.close()

    def test_template_applied_once(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return prompt_text(messages[0]["content"])

        tok = Tokenizer()
        self.assertEqual(template_prompt(tok, "raw", "hello"), "hello")
        self.assertEqual(template_prompt(tok, "project", "hello"), prompt_text("hello"))
        tok.apply_chat_template = lambda *args, **kwargs: prompt_text(
            prompt_text("hello")
        )
        with self.assertRaises(ValueError):
            template_prompt(tok, "project", "hello")

    def test_changed_protocol_requires_a_new_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            freeze(path, {"key": 1})
            freeze(path, {"key": 1})
            with self.assertRaises(ValueError):
                freeze(path, {"key": 2})

    def test_gold_index_follows_candidate_normalization_and_ifeval_uses_all(self):
        try:
            from nlp_course.smollm2_report import verify_row, paired
        except ModuleNotFoundError as exc:
            if exc.name == "numpy":
                self.skipTest(
                    "NumPy is required for the optional reporting environment"
                )
            raise
        prompt = dict(id="q", choices=[" X", " long"], gold=1)
        # Raw likelihood prefers choice 0. Per-character normalization prefers 1.
        record = dict(
            id="q",
            logprob=[-2.0, -3.0],
            normalized=[-2.0, -0.75],
            gold=1,
            pred=1,
            correct=True,
        )
        verify_row("piqa", record, prompt)
        with self.assertRaises(ValueError):
            verify_row("piqa", dict(record, pred=0), prompt)
        row = dict(
            id="q",
            correct=False,
            scores=dict(
                inst_level_strict_acc=[True, False],
                prompt_level_strict_acc=0,
                inst_level_loose_acc=[True, True],
                prompt_level_loose_acc=1,
            ),
        )
        src = dict(id="q", instructions=["a", "b"])
        verify_row("ifeval", row, src)
        with self.assertRaises(ValueError):
            verify_row("ifeval", dict(row, correct=True), src)
        other = dict(
            row, scores=dict(row["scores"], inst_level_strict_acc=[False, False])
        )
        diff = paired([row], [other], "inst_level_strict_acc", 42)
        self.assertEqual(diff["delta_pp"], 50.0)
        self.assertEqual(diff["ci95"], [50.0, 50.0])
