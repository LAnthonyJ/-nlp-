import math
import unittest

import torch

from nlp_course.progression_data import instruction_ok
from nlp_course.progression_engine import completion_tokens
from nlp_course.progression_v2 import needs_anchor
from nlp_course.progression_v2_data import compound
from nlp_course.training import preference_loss


class BoundaryTokenizer:
    eos_token_id = 0
    completion_v2 = True

    def encode(self, text, add_special_tokens=False):
        ids = list(text.encode())
        if text.startswith(":C"):
            ids[:2] = [259]
        return ids


class CompletionTests(unittest.TestCase):
    def test_supervised_span(self):
        tok = BoundaryTokenizer()
        for text in ("CORK", " leading", "\n", '["red"]'):
            p, a = completion_tokens(tok, ":", text, "ifeval")
            self.assertEqual(p, tok.encode(":"))
            self.assertEqual(a, tok.encode(text) + [0])
        for text in ("", "\x00", "X" * 2047):
            with self.assertRaises(ValueError):
                completion_tokens(tok, ":", text, "gsm8k")
        self.assertEqual(
            completion_tokens(tok, ":", "CORK", "piqa", False), ([58], list(b"ORK"))
        )

    def test_frozen_reference_and_loss(self):
        policy = torch.nn.Parameter(torch.tensor([-3.0, -5.0]))
        ref = policy.detach().clone()
        loss = preference_loss(policy[:1], policy[1:], ref[:1], ref[1:], 0.1)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        torch.optim.SGD([policy], lr=0.1).step()
        self.assertTrue(policy[0] > ref[0] and policy[1] < ref[1])
        self.assertEqual(ref.tolist(), [-3.0, -5.0])

    def test_multiconstraint_labels(self):
        row = {"id": "if-demo", "words": ["apple", "elm", "fir"]}
        for split in ("sft", "mine", "dev", "holdout"):
            item = compound(row, split, 0)
            self.assertTrue(instruction_ok(item, item["answer"]))
            self.assertFalse(instruction_ok(item, item["answer"].upper()))
            self.assertFalse(
                instruction_ok(item, ",".join(reversed(item["answer"].split(","))))
            )

    def test_hybrid_trigger(self):
        base = {"ifeval": 30, "gsm8k": 20}
        weak = [dict(base, chosen_delta=-i, rejected_delta=-3 * i) for i in (1, 2)]
        self.assertTrue(needs_anchor(weak, base))
        self.assertFalse(needs_anchor(weak[:1], base))
        self.assertFalse(needs_anchor([weak[0], dict(weak[1], gsm8k=21)], base))
