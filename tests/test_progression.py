import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path

from nlp_course.progression_data import (
    TASKS,
    check_splits,
    final_value,
    instruction,
    instruction_ok,
    qa,
)
from nlp_course.progression_engine import candidate_pair, grade, prompt_for
from nlp_course.progression_stats import compare, scores
from nlp_course.smollm2_data import freeze


class ProgressionTests(unittest.TestCase):
    def test_published_predictions_reproduce_full_scores(self):
        import gzip
        import json
        from nlp_course.capability import read, sha

        folder = Path(__file__).resolve().parents[1] / "docs/progression-results"
        for name, checksum in read(folder / "manifest.json").items():
            self.assertEqual(sha(folder / name), checksum)
        metrics = read(folder / "metrics.json")
        by_arm = {key: [] for key in ("base", "sft", "dpo")}
        with gzip.open(
            folder / "predictions.jsonl.gz", "rt", encoding="utf8"
        ) as stream:
            for line in stream:
                row = json.loads(line)
                by_arm[row["model"]].append(row)
        expected = metrics["public"]["scores"]
        ids = {r["id"] for r in by_arm["base"]}
        for arm, rows in by_arm.items():
            self.assertEqual({r["id"] for r in rows}, ids)
            self.assertEqual(scores(rows), expected[arm])
        for run in metrics["history"]:
            if run["opened"]:
                self.assertEqual(
                    run["passed"],
                    run["formal_sft"]["passed"] and run["formal_dpo"]["passed"],
                )

    def test_formal_cache_requires_same_weights_and_complete_batch(self):
        from nlp_course.capability import save
        from nlp_course.progression_report import reuse_formal_batches

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root / "opened/1.json", dict(attempt=1, base="revision", sft="old"))
            rows = [dict(id=str(i), task="piqa") for i in range(4)]
            for stage in ("base", "sft"):
                folder = root / f"round-1/formal-{stage}/piqa"
                save(folder / "0.json", rows[:2])
                save(folder / "2.json", rows[2:3])
            reuse_formal_batches(root, rows, 2, dict(base="revision", sft="new"))
            self.assertTrue((root / "public/base/piqa/0.json").exists())
            self.assertFalse((root / "public/base/piqa/2.json").exists())
            self.assertFalse((root / "public/sft/piqa/0.json").exists())

    def test_lr_retry_reuses_only_fixed_inputs(self):
        from nlp_course.capability import save, sha
        from nlp_course.progression import reuse_fixed_inputs

        with tempfile.TemporaryDirectory() as tmp:
            old, new = (Path(tmp) / s for s in ("round-2", "round-3"))
            checkpoint = old / "sft/step-1"
            checkpoint.mkdir(parents=True)
            weight = checkpoint / "adapter_model.safetensors"
            weight.write_bytes(b"fixture")
            save(checkpoint.parent / "selected.json", dict(weights=sha(weight)))
            design = dict(
                hard=True,
                dev=128,
                extra=None,
                sft_lr=0.0001,
                dpo_lr=0.00001,
                sft_parent=str(checkpoint),
            )
            save(old / "outcome.json", dict(action="dpo_lr", sft=str(checkpoint)))
            save(old / "design.json", design)
            save(old / "mining/pairs.json", [{"id": "x"}])
            save(new / "design.json", dict(design, hard=False, dpo_lr=0.000005))
            reuse_fixed_inputs(old, new)
            self.assertFalse((new / "mining/pairs.json").exists())
            save(new / "design.json", dict(design, dpo_lr=0.000005))
            reuse_fixed_inputs(old, new)
            reuse_fixed_inputs(old, new)
            self.assertEqual(
                sha(new / "mining/pairs.json"), sha(old / "mining/pairs.json")
            )

    def test_number_substitution_cannot_cross_panels_or_training(self):
        x = dict(id="public1", prompt="How many are left from 3 blue balls?")
        y = dict(id="training1", prompt="How many are left from 12 blue balls!")
        with self.assertRaises(ValueError):
            check_splits({"sft": [y]}, [[x], [], []])
        with self.assertRaises(ValueError):
            check_splits({}, [[y], [x], []])
        check_splits({}, [[x, y], [], []])

    def test_dev_gain_cannot_cover_a_large_single_task_loss(self):
        if find_spec("numpy") is None:
            self.skipTest("NumPy required")
        old = [
            dict(id=f"{t}-{i}", task=t, correct=(t == "piqa"))
            for t in TASKS
            for i in range(10)
        ]
        new = [dict(r, correct=r["task"] != "piqa") for r in old]
        result = compare(old, new, 2)
        self.assertGreater(result["pp"], 2)
        self.assertFalse(result["passed"])

    def test_numeric_claims_are_consistent(self):
        self.assertEqual(final_value("I add 5 and 7.\n#### 12\n12.0"), 12)
        self.assertEqual(
            final_value("After adding them:\n-2/3"), final_value("#### -4/6")
        )
        self.assertIsNone(final_value("#### 12\nFinal answer: 10"))
        self.assertIsNone(final_value("There were 9 apples and 3 pears."))
        self.assertIsNone(final_value("Final answer: ambiguous\n12"))

    def test_common_prompt_and_instruction_content(self):
        record = instruction("dev", 2, ["amber", "elm", "cedar"])
        self.assertEqual(prompt_for(record), qa(record["prompt"]))
        self.assertTrue(instruction_ok(record, '[ "amber", "elm", "cedar" ]'))
        self.assertFalse(instruction_ok(record, '["elm","cedar","amber"]'))
        self.assertEqual(
            prompt_for(dict(prompt="Context", choices=[" a", " b"])), "Context"
        )

    def test_mined_wrong_answer_and_label_direction(self):
        r = dict(task="gsm8k", gold="10", answer="#### 10")
        out = [
            dict(response=s, truncated=False) for s in ["#### 9", "nonsense", "#### 10"]
        ]
        pair = candidate_pair(r, out)
        self.assertTrue(pair["greedy_error"])
        self.assertTrue(grade(r, pair["chosen"]))
        self.assertFalse(grade(r, pair["rejected"]))
        self.assertEqual(pair["source"], "model")
        self.assertIsNone(candidate_pair(r, out[1:]))

    def test_same_ids_and_macro_gate(self):
        if find_spec("numpy") is None:
            self.skipTest("NumPy required for confidence intervals")
        a = [
            dict(id=f"{t}/{i}", task=t, correct=False)
            for t in TASKS
            for i in range(12 if t == "piqa" else 4)
        ]
        b = [dict(r, correct=r["task"] == "piqa") for r in a]
        self.assertEqual(scores(b)["mean"], 25)
        result = compare(a, list(reversed(b)), 2, True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["lower"], 25)
        c = [dict(r, correct=r["task"] == "ifeval") for r in a]
        self.assertFalse(compare(c, b, 1)["passed"])
        with self.assertRaises(ValueError):
            compare(a, b[1:], 1)

    def test_immutable_panel_resumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "opened.json"
            freeze(path, {"sft": "first", "dpo": "frozen"})
            freeze(path, {"sft": "first", "dpo": "frozen"})
            with self.assertRaises(ValueError):
                freeze(path, {"sft": "first", "dpo": "retuned"})
