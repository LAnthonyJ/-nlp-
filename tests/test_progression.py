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
