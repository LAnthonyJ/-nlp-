import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nlp_course.capability_data import prepare
from nlp_course.capability import (score_task, wilson, blind_review, import_review,
                                  analyze, save, read, score)


class CapabilityTests(unittest.TestCase):
    def test_numeric_and_format_failures_are_separate(self):
        task=dict(category='math',reference='2',id='a',split='main')
        row=dict(response='The answer is: 3',stage='base',finish_reason='eos')
        self.assertEqual(score_task(task,row)['status'],'incorrect')
        row['response']='3'
        self.assertEqual(score_task(task,row)['status'],'invalid_answer')
        row['response']='The answer is: 2';row['finish_reason']='length'
        self.assertTrue(score_task(task,row)['correct'])
        self.assertTrue(score_task(task,row)['truncated'])

    def test_unexecuted_code_is_not_incorrect(self):
        task=dict(category='code',id='a',split='main')
        row=dict(response='def f(): pass',stage='dpo',finish_reason='eos')
        self.assertIsNone(score_task(task,row)['correct'])

    def test_probability_intervals_at_boundaries(self):
        self.assertIsNone(wilson(0,0))
        for correct in [0,5,10]:
            lo,hi=wilson(correct,10)
            self.assertTrue(0<=lo<=correct/10<=hi<=1)

    def test_lexical_probe_is_not_factual_accuracy(self):
        task=dict(category='concept',id='a',split='diagnostic',keywords=['price','time'])
        row=dict(response='price',stage='sft',finish_reason='eos')
        result=score_task(task,row)
        self.assertIsNone(result['correct']);self.assertEqual(result['keyword_coverage'],.5)

    def test_freeze_counts_and_training_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'gsm.jsonl').write_text('\n'.join(json.dumps(dict(query=f'math question {i}',response='#### 1')) for i in range(1319)))
            (root/'code.jsonl').write_text('\n'.join(json.dumps(dict(task_id=f'H/{i}',prompt=f'def f{i}():',canonical_solution=' pass',test='',entry_point=f'f{i}')) for i in range(164)))
            for name in ['train_sft.json','train_dpo.json']:
                save(root/name,[dict(instruction=f'UNRELATED train {i}',output='x') for i in range(25)])
            report=prepare(root/'gsm.jsonl',root,root/'suite',root/'code.jsonl')
            tasks=report['tasks']
            self.assertEqual(len(tasks),140)
            self.assertEqual(len({t['id'] for t in tasks}),140)
            self.assertEqual(sum(t['split']=='main' for t in tasks),100)
            report2=prepare(root/'gsm.jsonl',root,root/'second',root/'code.jsonl')
            self.assertEqual(report,report2)
            save(root/'train_sft.json',[dict(instruction=tasks[0]['prompt']) for _ in range(25)])
            with self.assertRaisesRegex(ValueError,'overlap'):
                prepare(root/'gsm.jsonl',root,root/'third',root/'code.jsonl')

    def test_blinding_and_incomplete_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);suite=root/'suite';run=root/'run';(run/'formal').mkdir(parents=True)
            tasks=[dict(id=f'{c}-{i}',category=c,split='main',prompt='p',reference='r') for c in ['concept','writing'] for i in range(10)]
            save(suite/'suite.json',dict(tasks=tasks))
            for stage in ['base','sft','dpo']:
                (run/'formal'/f'{stage}.jsonl').write_text('\n'.join(json.dumps(dict(task_id=t['id'],stage=stage,response='answer')) for t in tasks))
            blind_review(suite,run)
            key=read(run/'private-review-key.json')
            self.assertEqual(len(key),66)
            self.assertEqual(sum(r['is_repeat'] for r in key.values()),6)
            with self.assertRaises(ValueError):import_review(run,run/'blind-review.csv')
            with (run/'blind-review.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
            for row in rows:
                for field in ['accuracy_or_completion','completeness_or_coherence','clarity_or_expression']:row[field]='3'
            with (run/'rated.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            import_review(run,run/'rated.csv')
            save(run/'scores.json',dict(records=[]))
            report=analyze(suite,run)
            self.assertEqual(report['human_status'],'scored')
            self.assertEqual(report['human_repeat_mean_difference'],0)
            self.assertTrue(all(r['n']==10 for r in report['human_summary']))

    def test_missing_predictions_remain_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            save(root/'suite/suite.json',dict(tasks=[dict(id='a',category='math',split='main')]))
            score(root/'suite',root/'run')
            report=analyze(root/'suite',root/'run')
            self.assertIsNone(report['summary'][0]['accuracy'])
            self.assertEqual(report['human_status'],'pending')


if __name__=='__main__':unittest.main()
