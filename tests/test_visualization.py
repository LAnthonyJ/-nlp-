import copy
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import torch
from nlp_course.demo import run_cpu_demo
from nlp_course.evaluation import score_predictions
from nlp_course.visualization import collect, load_reports, main, render, validate_reports, write_json


class VisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = run_cpu_demo(include_trace=True)
        cls.root = Path(__file__).resolve().parents[1]
        cls.fixture = cls.root / 'examples/toy_predictions.jsonl'
        rows = [json.loads(line) for line in cls.fixture.read_text().splitlines()]
        cls.evaluation = score_predictions([r['prediction'] for r in rows], [r['reference'] for r in rows])

    def test_default_report_preserves_original_interface_and_values(self):
        default = run_cpu_demo()
        expected = {k: v for k, v in self.demo.items() if k != 'trace'}
        self.assertEqual(default, expected)
        self.assertAlmostEqual(default['lora_synthetic_loss_before'], .04594896361231804, places=6)
        self.assertEqual(default['transformer_shape'], [2, 5, 16])

    def test_original_script_still_prints_only_json(self):
        output = subprocess.run([sys.executable, str(self.root / 'examples/cpu_demo.py')],
                                capture_output=True, text=True, check=True).stdout
        self.assertNotIn('trace', json.loads(output))
        self.assertEqual(json.loads(output)['seed'], 42)

    def test_trace_covers_actual_steps_and_merge_outputs(self):
        trace = self.demo['trace']
        self.assertEqual([p['step'] for p in trace['loss_history']], list(range(81)))
        self.assertEqual(trace['loss_history'][0]['mse'], self.demo['lora_synthetic_loss_before'])
        self.assertEqual(trace['loss_history'][-1]['mse'], self.demo['lora_synthetic_loss_after'])
        before, after = [torch.tensor(trace['merge_outputs'][key]) for key in ('before', 'after')]
        self.assertEqual(before.shape, (64, 3))
        self.assertEqual((before - after).abs().max().item(), self.demo['lora_merge_max_error'])
        torch.testing.assert_close(before, after)

    def test_collect_keeps_four_sample_report_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            collect(path, self.fixture)
            demo, evaluation = load_reports(path)
            self.assertEqual(demo['toy_evaluation']['total'], 2)
            self.assertEqual((evaluation['total'], evaluation['correct'], evaluation['invalid_outputs']), (4, 2, 1))
            self.assertTrue((path / 'manifest.json').is_file())

    def test_invalid_or_incomplete_data_is_rejected(self):
        mutations = [
            lambda d, e: d.pop('trace'),
            lambda d, e: d['trace']['loss_history'].pop(),
            lambda d, e: d['trace']['loss_history'][0].update(mse=float('nan')),
            lambda d, e: d['trace']['loss_history'][0].update(mse=0),
            lambda d, e: d.update(lora_synthetic_loss_after=1),
            lambda d, e: d['trace']['merge_outputs']['after'].pop(),
            lambda d, e: d.update(lora_merge_max_error=1),
            lambda d, e: d['sampling_distributions'].update(topk=[-.1, .6, .5]),
            lambda d, e: d['sampling_distributions'].update(topk=[.1, .1, .1]),
            lambda d, e: d['sampling_distributions'].update(topk=[1., 0., 0.]),
            lambda d, e: d['trace'].update(original_distribution=[1., 0., 0.]),
            lambda d, e: e.update(correct=3),
        ]
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                demo, evaluation = copy.deepcopy(self.demo), copy.deepcopy(self.evaluation)
                mutate(demo, evaluation)
                with self.assertRaises(ValueError):
                    validate_reports(demo, evaluation)

    def test_missing_reports_fail_before_rendering(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                render(Path(directory), Path(directory) / 'figures')
            with patch('sys.stderr'):
                with self.assertRaises(SystemExit) as result:
                    main(['render', '--data-dir', directory])
                self.assertEqual(result.exception.code, 2)

    def test_core_import_does_not_import_matplotlib(self):
        code = "import sys; import nlp_course.demo, nlp_course.visualization; assert 'matplotlib' not in sys.modules"
        subprocess.run([sys.executable, '-c', code], check=True)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Install the viz extra to test figure export')
    def test_exports_ten_valid_files_without_rerunning_training(self):
        with tempfile.TemporaryDirectory() as directory:
            data, output = Path(directory) / 'data', Path(directory) / 'figures'
            data.mkdir()
            write_json(data / 'cpu-demo.json', self.demo)
            write_json(data / 'toy-evaluation.json', self.evaluation)
            before = {p.name: p.read_bytes() for p in data.iterdir()}
            with patch('nlp_course.demo.run_cpu_demo', side_effect=AssertionError('Rendering must not train')):
                files = render(data, output)
            self.assertEqual(len(files), 10)
            for path in files:
                self.assertGreater(path.stat().st_size, 1000)
                if path.suffix == '.png':
                    expected = (2160, 1440) if path.stem == 'overview' else (1440, 810)
                    self.assertEqual(struct.unpack('>II', path.read_bytes()[16:24]), expected)
                else:
                    self.assertTrue(ET.parse(path).getroot().tag.endswith('svg'))
            self.assertEqual(before, {p.name: p.read_bytes() for p in data.iterdir()})


if __name__ == '__main__':
    unittest.main()
