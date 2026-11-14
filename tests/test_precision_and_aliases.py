"""Regressions for precision and shared-module findings against 20bfd8b."""
import unittest
import torch
from torch import nn

from nlp_course.decoding import DecodingSampler
from nlp_course.lora import LoRALinear, inject_lora_modules, mark_only_lora_as_trainable


class RankingPrecisionTests(unittest.TestCase):
    def test_greedy_uses_logit_order_before_probability_rounding(self):
        sampler = DecodingSampler()
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for temperature in [1., 1e10, 1e300]:
                with self.subTest(dtype=dtype, temperature=temperature):
                    logits = torch.tensor([0., 1e-5], dtype=dtype)
                    probs = sampler.distribution(logits, temperature=temperature)
                    self.assertEqual(probs.dtype, dtype)
                    self.assertEqual(probs.tolist(), [0., 1.])
                    self.assertEqual(sampler.sample(logits, temperature=temperature), 1)

    def test_topk_preserves_support_at_high_temperatures(self):
        for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
            for temperature in [1e10, 1e300]:
                with self.subTest(dtype=dtype, temperature=temperature):
                    pair = torch.tensor([0., 1e-5], dtype=dtype)
                    selected = DecodingSampler().distribution(pair, strategy='topk', top_k=1,
                                                             temperature=temperature)
                    self.assertEqual(selected.tolist(), [0., 1.])
                    logits = torch.tensor([3., 2., 0., 1.], dtype=dtype)
                    probs = DecodingSampler().distribution(logits, strategy='topk', top_k=2,
                                                          temperature=temperature)
                    self.assertEqual(probs.dtype, dtype)
                    self.assertEqual(probs.nonzero().flatten().tolist(), [0, 1])
                    expected = torch.zeros_like(probs)
                    expected[:2] = torch.softmax(logits[:2].double() / temperature, 0).to(dtype)
                    torch.testing.assert_close(probs, expected)


class SharedModuleInjectionTests(unittest.TestCase):
    def check_aliases(self, names):
        shared = nn.Linear(4, 4)
        model = nn.ModuleDict({name: shared for name in names}).eval()
        x = torch.randn(3, 4)
        expected = shared(x).detach()
        self.assertEqual(inject_lora_modules(model, r=2, alpha=2), 1)
        adapter = model[names[0]]
        self.assertIsInstance(adapter, LoRALinear)
        self.assertFalse(adapter.training)
        for name in names:
            self.assertIs(model[name], adapter)
            torch.testing.assert_close(model[name](x), expected)
        with torch.no_grad():
            adapter.lora_B.weight.fill_(0.5)
        for name in names:
            torch.testing.assert_close(model[name](x), adapter(x))
        self.assertEqual(inject_lora_modules(model, r=2), 0)
        self.assertEqual(mark_only_lora_as_trainable(model), 16)
        adapter(x).square().sum().backward()
        self.assertIsNotNone(adapter.lora_B.weight.grad)
        expected = adapter(x).detach()
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.merge()
        for name in names:
            torch.testing.assert_close(model[name](x), expected)

    def test_two_target_names_share_one_adapter(self):
        self.check_aliases(['q_proj', 'v_proj'])

    def test_untargeted_alias_registered_first_is_preserved(self):
        self.check_aliases(['other', 'q_proj'])

    def test_untargeted_alias_registered_last_is_preserved(self):
        self.check_aliases(['q_proj', 'other'])

    def test_nested_aliases_of_shared_parent_are_preserved(self):
        shared = nn.Linear(4, 4)
        block = nn.ModuleDict({'q_proj': shared, 'other': shared})
        model = nn.ModuleDict({'left': block, 'right': block, 'elsewhere': shared})
        self.assertEqual(inject_lora_modules(model, r=2), 1)
        self.assertIs(model['left'], model['right'])
        adapter = model['left']['q_proj']
        self.assertIs(adapter, model['left']['other'])
        self.assertIs(adapter, model['right']['q_proj'])
        self.assertIs(adapter, model['elsewhere'])
        self.assertEqual(inject_lora_modules(model, r=2), 0)
