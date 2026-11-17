import json
from pathlib import Path
import tempfile
import unittest

import torch

from nlp_course.training import encode, preference_loss, prepare
from nlp_course.experiment import repair_tail
from nlp_course.capability import save, read, sha


class Tokenizer:
    unk_token_id = -1
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]
    def convert_tokens_to_ids(self, token):
        return 1000


class TrainingTests(unittest.TestCase):
    def test_published_repair_statistics_and_file_hashes_are_consistent(self):
        root=Path(__file__).resolve().parents[1]/'docs/repair-results'
        for name,expected in read(root/'manifest.json')['files'].items():
            self.assertEqual(sha(root/name),expected,name)
        config=read(root/'experiment.json')
        tasks=read(root/'suite/suite.json')['tasks']
        records=read(root/'scores.json')['records']
        self.assertEqual(len(records),len(config['models'])*len(tasks))
        self.assertEqual(len({(r['stage'],r['task_id']) for r in records}),len(records))
        for summary in read(root/'analysis.json')['summary']:
            rows=[r for r in records if r['stage']==summary['model'] and r['category']==summary['category']]
            self.assertEqual(summary['n'],len(rows))
            self.assertEqual(summary['correct'],sum(r['correct'] is True for r in rows))

    def test_only_completion_and_single_end_token_are_supervised(self):
        example=encode(Tokenizer(),'Question','Answer',500)
        supervised=[v for v in example['labels'] if v!=-100]
        self.assertEqual(supervised,[ord(c) for c in 'Answer']+[1000])
        self.assertEqual(example['input_ids'][-len(supervised):],supervised)
        for answer in ['', 'bad <|im_end|>']:
            with self.assertRaises(ValueError):encode(Tokenizer(),'Question',answer,500)
        with self.assertRaises(ValueError):encode(Tokenizer(),'Question','Answer',5)

    def test_preference_gradient_direction_and_reference_stays_constant(self):
        a=torch.tensor([1.],requires_grad=True)
        b=torch.tensor([1.],requires_grad=True)
        reference=torch.tensor([1.])
        preference_loss(a,b,reference,reference,.1).backward()
        self.assertLess(a.grad.item(),0);self.assertGreater(b.grad.item(),0)
        self.assertEqual(reference.item(),1)
        self.assertAlmostEqual(preference_loss(a,b,reference,reference,.1).item(),.693147,places=5)
        a.grad=None;b.grad=None
        preference_loss(b,a,reference,reference,.1).backward()
        self.assertGreater(a.grad.item(),0)

    def test_native_end_token_is_used_for_a_base_model(self):
        tokenizer=Tokenizer();tokenizer.eos_token_id=37
        example=encode(tokenizer,'Question','Answer',500)
        self.assertEqual(example['labels'][-1],37)
        self.assertNotIn(1000,example['labels'])

    def test_resume_repairs_only_unterminated_tail(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'records.jsonl'
            p.write_bytes(b'{"a":1}\n{"broken":')
            repair_tail(p)
            self.assertEqual(p.read_bytes(),b'{"a":1}\n')
            p.write_bytes(b'{"a":1}\nnot-json\n')
            with self.assertRaises(ValueError):repair_tail(p)

    def test_unreviewed_course_data_is_quarantined_and_split_groups_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            save(root/'train_sft.json',[dict(instruction='x',output='false answer')])
            save(root/'train_dpo.json',[dict(instruction='x',chosen='a',rejected='a')])
            prepare(dict(train_dir=str(root),data_dir=str(root/'prepared')))
            audit=read(root/'prepared/audit.json')
            self.assertEqual(audit[0]['reason'],'needs_content_review')
            self.assertEqual(audit[1]['reason'],'identical_pair')
            train=read(root/'prepared/sft_train.json');val=read(root/'prepared/sft_validation.json')
            self.assertTrue(train and val)
            self.assertFalse({r['group'] for r in train}&{r['group'] for r in val})
            self.assertTrue(all(r['source']!='train_dpo.json' for r in read(root/'prepared/dpo_train.json')))


if __name__=='__main__':unittest.main()
