import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from torch import nn

from nlp_course.transformer import build_transformer, causal_mask, LayerNorm, MultiHeadAttention, PositionalEncoding
from nlp_course.lora import LoRALinear, inject_lora_modules, mark_only_lora_as_trainable, count_model_params
from nlp_course.decoding import DecodingSampler
from nlp_course.evaluation import batch_data, extract_final_answer, extract_answer_number, score_predictions, evaluate_generated, main


torch.set_num_threads(1)


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = build_transformer(12, 12, 6, 6, embed_dim=8, N=2, heads=2, hidden_dim=16, dropout=0)

    def test_layers_have_independent_parameters(self):
        for stack in [self.model.encoder, self.model.decoder]:
            first, second = stack.layers
            self.assertIsNot(first, second)
            for a, b in zip(first.parameters(), second.parameters()):
                self.assertNotEqual(a.data_ptr(), b.data_ptr())

    def test_normalization_matches_reference(self):
        norm = LayerNorm(4).double()
        x = torch.randn(2, 3, 4, dtype=torch.float64)
        torch.testing.assert_close(norm(x), nn.LayerNorm(4).double()(x))
        self.assertTrue(torch.isfinite(norm(torch.ones_like(x))).all())

    def test_odd_positional_width_and_capacity(self):
        pe = PositionalEncoding(5, 4, 0)
        self.assertEqual(pe(torch.zeros(2, 4, 5)).shape, (2, 4, 5))
        with self.assertRaises(ValueError):
            pe(torch.zeros(2, 5, 5))

    def test_future_tokens_do_not_leak(self):
        model = self.model.eval()
        src = torch.randint(0, 12, (2, 5))
        tgt = torch.randint(0, 12, (2, 5))
        changed = tgt.clone()
        changed[:, 3:] = (changed[:, 3:] + 1) % 12
        enc = model.encode(src, None)
        a = model.decode(tgt, enc, None, causal_mask(5))
        b = model.decode(changed, enc, None, causal_mask(5))
        torch.testing.assert_close(a[:, :3], b[:, :3])

    def test_padding_blocks_changed_key_values(self):
        attn = MultiHeadAttention(8, 2, 0).eval()
        q, k = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool).expand(2, -1)
        changed = k.clone()
        changed[:, 3:] += 100
        torch.testing.assert_close(attn(q, k, k, mask), attn(q, changed, changed, mask))

    def test_mask_shapes_and_all_masked_backward(self):
        attn = MultiHeadAttention(8, 2, 0)
        x = torch.randn(2, 3, 8, requires_grad=True)
        mask3 = causal_mask(3).expand(2, -1, -1)
        torch.testing.assert_close(attn(x, x, x, mask3), attn(x, x, x, mask3[:, None]))
        out = attn(x, x, x, torch.zeros(2, 3, dtype=torch.bool))
        out.sum().backward()
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.isfinite(x.grad).all())
        for bad in [torch.ones(2, 3), torch.ones(2, 4, dtype=torch.bool)]:
            with self.assertRaises(ValueError):
                attn(x, x, x, bad)

    def test_end_to_end_backward_and_probability(self):
        ids = torch.randint(0, 12, (2, 5))
        out = self.model.project(self.model.decode(ids, self.model.encode(ids, None), None, causal_mask(5)))
        torch.testing.assert_close(out.exp().sum(-1), torch.ones(2, 5))
        nn.functional.nll_loss(out.reshape(-1, 12), ids.reshape(-1)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()))


class LoRATests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)

    def test_initial_zero_update_dtype_and_base_freezing(self):
        base = nn.Linear(4, 3).double()
        x = torch.randn(2, 4, dtype=torch.float64)
        expected = base(x).detach()
        layer = LoRALinear(base, r=2)
        torch.testing.assert_close(layer(x), expected)
        self.assertEqual(layer.lora_A.weight.dtype, torch.float64)
        layer(x).square().sum().backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNone(base.bias.grad)
        self.assertIsNotNone(layer.lora_A.weight.grad)
        self.assertGreater(layer.lora_B.weight.grad.abs().sum().item(), 0)

    def test_merge_equivalence_idempotence_and_roundtrip(self):
        layer = LoRALinear(nn.Linear(4, 3), r=2, dropout=0.5).eval()
        nn.init.normal_(layer.lora_B.weight)
        x = torch.randn(6, 4)
        expected = layer(x).detach()
        layer.merge()
        torch.testing.assert_close(layer(x), expected)
        state = copy.deepcopy(layer.state_dict())
        layer.merge()
        for key, tensor in state.items():
            torch.testing.assert_close(layer.state_dict()[key], tensor)
        restored = LoRALinear(nn.Linear(4, 3), r=2, dropout=0.5)
        restored.load_state_dict(state)
        torch.testing.assert_close(restored.eval()(x), expected)
        with self.assertRaises(RuntimeError):
            restored.train()

    def test_merge_requires_eval_and_invalid_parameters(self):
        with self.assertRaises(RuntimeError):
            LoRALinear(nn.Linear(2, 2)).merge()
        for kwargs in [dict(r=0), dict(r=-1), dict(r=True), dict(alpha=0), dict(dropout=1), dict(alpha=float('nan'))]:
            with self.assertRaises(ValueError):
                LoRALinear(nn.Linear(2, 2), **kwargs)

    def test_injection_nested_targets_and_freezing(self):
        model = nn.ModuleDict({'block': nn.ModuleDict({'q_proj': nn.Linear(4, 4)}), 'other': nn.Linear(4, 4)})
        model.eval()
        self.assertEqual(inject_lora_modules(model, r=2), 1)
        self.assertFalse(model['block']['q_proj'].training)
        self.assertEqual(inject_lora_modules(model, r=2), 0)
        self.assertEqual(mark_only_lora_as_trainable(model), 16)
        self.assertEqual(count_model_params(model)[0], 16)
        self.assertFalse(model['other'].weight.requires_grad)
        self.assertEqual(count_model_params(nn.Identity()), (0, 0, 0.0))


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return 'prompt'

    def __call__(self, *args, **kwargs):
        return {'input_ids': torch.tensor([[0, 1]]), 'attention_mask': torch.tensor([[0, 1]])}

    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids.tolist()))


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.seen_masks = []

    def forward(self, input_ids, attention_mask):
        assert not self.training
        self.seen_masks.append(attention_mask.clone())
        logits = torch.tensor([0., 1., 9.]).expand(1, input_ids.shape[1], 3)
        return SimpleNamespace(logits=logits)


class DecodingTests(unittest.TestCase):
    def setUp(self):
        self.sampler = DecodingSampler(eos_token_id=2)
        self.logits = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64).log()

    def test_greedy_and_topk_support(self):
        torch.testing.assert_close(self.sampler.distribution(self.logits), torch.tensor([1., 0., 0.], dtype=torch.float64))
        p = self.sampler.distribution(self.logits, strategy='topk', top_k=2)
        torch.testing.assert_close(p, torch.tensor([2/3, 1/3, 0.], dtype=torch.float64))

    def test_topp_keeps_crossing_token(self):
        p = self.sampler.distribution(self.logits, strategy='topp', top_p=0.8)
        torch.testing.assert_close(p, torch.tensor([2/3, 1/3, 0.], dtype=torch.float64))
        self.assertEqual(self.sampler.distribution(self.logits, strategy='topp', top_p=0.1)[0], 1)

    def test_relative_minp_and_boundaries(self):
        p = self.sampler.distribution(self.logits, strategy='minp', min_p=0.4)
        torch.testing.assert_close(p, torch.tensor([2/3, 1/3, 0.], dtype=torch.float64))
        torch.testing.assert_close(self.sampler.distribution(self.logits, strategy='minp', min_p=0), self.logits.exp())
        self.assertEqual(self.sampler.distribution(self.logits, strategy='minp', min_p=1)[0], 1)

    def test_reproducible_sampling(self):
        a, b = torch.Generator().manual_seed(12), torch.Generator().manual_seed(12)
        kwargs = dict(strategy='topk', top_k=3)
        self.assertEqual([self.sampler.sample(self.logits, generator=a, **kwargs) for _ in range(20)],
                         [self.sampler.sample(self.logits, generator=b, **kwargs) for _ in range(20)])

    def test_repetition_penalty_applies_once_per_token(self):
        logits = torch.tensor([-2., 4., 3.])
        actual = self.sampler.penalize_repetition(logits, torch.tensor([0, 0, 1]), 2)
        torch.testing.assert_close(actual, torch.tensor([-4., 2., 3.]))
        torch.testing.assert_close(logits, torch.tensor([-2., 4., 3.]))

    def test_invalid_arguments(self):
        for kwargs in [dict(strategy='bad'), dict(temperature=0), dict(temperature=float('nan')),
                       dict(strategy='topk', top_k=0), dict(strategy='topp', top_p=0), dict(strategy='minp', min_p=1.1)]:
            with self.assertRaises(ValueError):
                self.sampler.distribution(self.logits, **kwargs)
        for logits in [torch.tensor([float('nan')]), torch.tensor([-float('inf')]), torch.tensor([float('inf')]), torch.zeros(2, 3)]:
            with self.assertRaises(ValueError):
                self.sampler.distribution(logits)

    def test_generate_default_eos_attention_and_mode_restore(self):
        model = FakeModel()
        self.assertEqual(self.sampler.generate(model, FakeTokenizer(), []), '2')
        self.assertTrue(model.training)
        self.assertEqual(len(model.seen_masks), 1)
        self.assertEqual(model.seen_masks[0].tolist(), [[0, 1]])

    def test_generate_all_strategies_and_mask_growth(self):
        for strategy in ['greedy', 'topk', 'topp', 'minp']:
            model = FakeModel().eval()
            result = self.sampler.generate(model, FakeTokenizer(), [], max_new_tokens=2,
                                           strategy=strategy, top_k=1, stop_on_eos=False)
            self.assertEqual(len(result.split()), 2)
            self.assertEqual(model.seen_masks[1].tolist(), [[0, 1, 1]])
            self.assertFalse(model.training)

    def test_generate_failure_restores_mode(self):
        model = FakeModel()
        with patch.object(model, 'forward', side_effect=RuntimeError('backend error')):
            with self.assertRaises(RuntimeError):
                self.sampler.generate(model, FakeTokenizer(), [])
        self.assertTrue(model.training)


class EvaluationTests(unittest.TestCase):
    def test_batches_cover_every_sample_once(self):
        for n in [0, 1, 2, 4, 5, 9]:
            for size in [1, 3, 10]:
                batches = batch_data(list(range(n)), size)
                self.assertEqual([x for b in batches for x in b], list(range(n)))
                self.assertTrue(all(0 < len(b) <= size for b in batches))
        with self.assertRaises(ValueError):
            batch_data([], 0)

    def test_numeric_extraction_no_rounding(self):
        for text, expected in [('1,234', 1234), ('-2', -2), ('1/2', 0.5), ('3.4', None), ('1/0', None), ('12,34', None)]:
            number = extract_answer_number('The answer is: ' + text)
            if text == '3.4':
                self.assertNotEqual(number, 3)
            elif expected is None:
                self.assertIsNone(number)
            else:
                self.assertEqual(number, expected)

    def test_boxed_nested_and_malformed(self):
        self.assertEqual(extract_final_answer(r'work \boxed{\frac{1}{2}}'), r'\frac{1}{2}')
        self.assertEqual(extract_answer_number(r'work \boxed{\frac{1}{2}}'), 0.5)
        self.assertIsNone(extract_final_answer(r'work \boxed{1'))
        self.assertIsNone(extract_answer_number('unmarked 42'))
        self.assertEqual(extract_answer_number('#### -1.5'), -1.5)

    def test_scoring_counts_invalid_and_fraction(self):
        result = score_predictions(['The answer is: 1/2', 'The answer is: 3.4', 'missing'], ['0.5', '3', '2'])
        self.assertEqual(result['correct'], 1)
        self.assertEqual(result['invalid_outputs'], 1)
        self.assertEqual(result['total'], 3)
        self.assertAlmostEqual(result['accuracy'], 1/3)

    def test_strict_math_does_not_claim_symbolic_equivalence(self):
        result = score_predictions([r'\boxed{x+x}', r'\boxed{\frac{1}{2}}'], ['2*x', '0.5'], 'math-strict')
        self.assertEqual([r['correct'] for r in result['records']], [False, True])

    def test_empty_mismatch_and_invalid_reference(self):
        self.assertIsNone(score_predictions([], [])['accuracy'])
        for predictions, refs in [(['a'], []), (['a'], ['bad']), (['a'], [r'\boxed{'])]:
            with self.assertRaises(ValueError):
                score_predictions(predictions, refs)

    def test_batched_backend_coverage_and_output_count(self):
        calls = []
        def backend(batch):
            calls.append(list(batch))
            return ['The answer is: ' + x for x in batch]
        result = evaluate_generated(['1', '2', '3', '4', '5'], ['1', '2', '3', '4', '5'], backend, 2)
        self.assertEqual(calls, [['1', '2'], ['3', '4'], ['5']])
        self.assertEqual(result['accuracy'], 1)
        with self.assertRaises(ValueError):
            evaluate_generated(['1', '2'], ['1', '2'], lambda b: [], 2)

    def test_cli_writes_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'predictions.jsonl', root / 'report.json'
            source.write_text(json.dumps(dict(prediction='#### 2', reference='2')) + '\n', encoding='utf-8')
            with patch('builtins.print'):
                main([str(source), '--output', str(output)])
            self.assertEqual(json.loads(output.read_text())['accuracy'], 1)

    def test_source_snapshots_unchanged(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / 'docs/source-manifest.json').read_text(encoding='utf-8-sig'))
        for entry in manifest:
            self.assertEqual(hashlib.sha256((root / entry['destination']).read_bytes()).hexdigest(), entry['sha256'])


if __name__ == '__main__':
    unittest.main()
