"""Regressions for the three P2 findings against commit 1c513ae."""
import unittest
import torch

from nlp_course.decoding import DecodingSampler
from nlp_course.evaluation import extract_final_answer, score_predictions
from nlp_course.transformer import MultiHeadAttention


class BoxedAnswerRegressions(unittest.TestCase):
    def test_nested_box_is_not_a_standalone_final_answer(self):
        prediction = r"\boxed{1 + \boxed{2}}"
        self.assertEqual(extract_final_answer(prediction), r"1 + \boxed{2}")
        self.assertEqual(score_predictions([prediction], ["2"], "math-strict")["correct"], 0)
        self.assertEqual(score_predictions([prediction], ["2"])["invalid_outputs"], 1)

    def test_unclosed_outer_box_does_not_accept_closed_inner_box(self):
        self.assertIsNone(extract_final_answer(r"\boxed{1 + \boxed{2}"))

    def test_last_independent_marker_still_wins(self):
        for suffix in [r"\boxed{3}", "The answer is: 3", "#### 3"]:
            with self.subTest(suffix=suffix):
                self.assertEqual(extract_final_answer(r"\boxed{1+\boxed{2}} then " + suffix), "3")

    def test_markers_inside_box_remain_part_of_outer_answer(self):
        self.assertEqual(extract_final_answer(r"\boxed{x + \text{The answer is: 2}}"),
                         r"x + \text{The answer is: 2}")


class TemperatureRegressions(unittest.TestCase):
    def test_extreme_temperature_produces_valid_distributions(self):
        sampler = DecodingSampler()
        cases = [
            (torch.tensor([60000., 60000., -60000.], dtype=torch.float16), 1e-5),
            (torch.tensor([1e30, 1e30, -1e30]), 1e-20),
            (torch.tensor([1., 1., -float('inf')]), 1e-300),
        ]
        for logits, temperature in cases:
            for strategy in ["greedy", "topk", "topp", "minp"]:
                with self.subTest(dtype=logits.dtype, temperature=temperature, strategy=strategy):
                    probs = sampler.distribution(logits, strategy=strategy, temperature=temperature, top_k=2)
                    self.assertTrue(torch.isfinite(probs).all())
                    self.assertTrue((probs >= 0).all())
                    self.assertAlmostEqual(probs.sum().item(), 1)
                    self.assertEqual(probs[-1].item(), 0)
                    self.assertIn(sampler.sample(logits, strategy=strategy, temperature=temperature, top_k=2), [0, 1])

    def test_large_temperature_does_not_overflow_subtraction(self):
        logits = torch.tensor([-1e308, 1e308], dtype=torch.float64)
        probs = DecodingSampler().distribution(logits, strategy="topk", top_k=2, temperature=1e308)
        torch.testing.assert_close(probs, torch.softmax(torch.tensor([-1., 1.], dtype=torch.float64), 0))


class AttentionShapeRegressions(unittest.TestCase):
    def setUp(self):
        self.attention = MultiHeadAttention(8, 2, 0)

    def test_mismatched_batches_are_rejected_before_projection(self):
        q, k, v = torch.randn(2, 3, 8), torch.randn(1, 6, 8), torch.randn(1, 6, 8)
        for inputs in [(q, k, v), (k, q, q), (q, q, v)]:
            with self.subTest(shapes=[tuple(x.shape) for x in inputs]):
                with self.assertRaisesRegex(ValueError, "batch"):
                    self.attention(*inputs)

    def test_invalid_rank_features_and_key_value_length(self):
        q = torch.randn(2, 3, 8)
        for inputs in [(q[0], q, q), (q, torch.randn(2, 4, 8), q), (q, q, torch.randn(2, 3, 4))]:
            with self.subTest(shapes=[tuple(x.shape) for x in inputs]):
                with self.assertRaises(ValueError):
                    self.attention(*inputs)

    def test_cross_attention_allows_different_query_and_key_lengths(self):
        q, kv = torch.randn(2, 3, 8, requires_grad=True), torch.randn(2, 5, 8, requires_grad=True)
        result = self.attention(q, kv, kv)
        self.assertEqual(result.shape, q.shape)
        result.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        self.assertTrue(torch.isfinite(kv.grad).all())
