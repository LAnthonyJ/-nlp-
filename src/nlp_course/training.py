"""Local, completion-only LoRA training with inspectable data and checkpoints."""
import argparse
import hashlib
import importlib.metadata
import json
import random
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .capability import read, save, sha

TEMPLATE = "{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


def prompt_text(prompt):
    return '<|im_start|>user\n' + prompt + '<|im_end|>\n<|im_start|>assistant\n'


def encode(tokenizer, prompt, answer, max_length):
    if not prompt.strip() or not answer.strip():
        raise ValueError('Empty prompt or completion')
    prefix = tokenizer.encode(prompt_text(prompt), add_special_tokens=False)
    completion = tokenizer.encode(answer, add_special_tokens=False)
    eos = getattr(tokenizer,'eos_token_id',None)
    if eos is None: eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if eos is None or eos == tokenizer.unk_token_id:
        raise ValueError('Tokenizer lacks ChatML end token')
    # Literal role delimiters in a completion can change its meaning.
    if '<|im_' in answer:
        raise ValueError('Completion contains role delimiter')
    ids = prefix + completion + [eos]
    if len(ids) > max_length:
        raise ValueError('Full completion exceeds sequence budget')
    return dict(input_ids=ids, labels=[-100] * len(prefix) + completion + [eos])


def group_key(prompt):
    text = re.sub(r'\d+', '#', prompt.casefold())
    return re.sub(r'\s+', ' ', text).strip()


def synthetic_rows():
    rows = []
    math_templates = ['Compute {a} * {b}.', 'What is the product of {a} and {b}?', 'Multiply {a} by {b}.', 'Evaluate {a} times {b}.', 'Find {a} multiplied by {b}.']
    for a in range(2, 18):
        for b in range(2, 10):
            rows.append(dict(prompt=math_templates[a%5].format(a=a,b=b)+' End with The answer is: followed by the result.', answer=f'The answer is: {a*b}', category='math', group=f'multiply-{a % 5}', source='integer oracle'))
    for i in range(120):
        yes = i % 2 == 0
        colors=[('amber','blue','green'),('red','warm','black'),('white','bright','yellow'),('violet','dark','silver'),('bronze','heavy','pink')][i%5]
        first,second,other=colors
        rows.append(dict(prompt=f'All {first} objects are {second}. Item {i} is {first if yes else other}. Must it be {second}? Answer yes or no only.', answer='yes' if yes else 'no', category='logic', group=f'logic-{i % 5}', source='logical entailment oracle'))
    operations = [('double', '* 2'), ('increment', '+ 1'), ('square', '** 2'), ('negate', '* -1'), ('halve', '/ 2')]
    for name, operation in operations:
        for i in range(16):
            function = f'{name}_{i}'
            rows.append(dict(prompt=f'Return a Python function {function}(x) that computes x {operation}. Return one complete Python code block.', answer=f'```python\ndef {function}(x):\n    return x {operation}\n```', category='code', group=f'code-{name}', source='arithmetic function oracle'))
    return rows


def prepare(config):
    out = Path(config['data_dir'])
    if (out / 'manifest.json').exists():
        raise ValueError('Prepared data already exists; use a new data directory')
    raw = Path(config['train_dir'])
    reviews = read(config['reviews']) if config.get('reviews') else {}
    suites = [read(path)['tasks'] for path in config.get('exclude_suites', [])]
    excluded = [t['prompt'] for suite in suites for t in suite]
    tokens = [set(p.casefold().split()) for p in excluded]
    rows, pairs, audit = [], [], []
    for name, target in [('train_sft.json', rows), ('train_dpo.json', pairs)]:
        for index, item in enumerate(read(raw / name)):
            record_id = hashlib.sha256(json.dumps(item, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            prompt = item.get('instruction', '')
            if isinstance(prompt,str) and item.get('input'):
                prompt += '\n' + item['input']
            reason = None
            if not isinstance(prompt, str) or not prompt.strip():
                audit.append(dict(id=record_id,file=name,index=index,reason='missing_prompt',meta_instruction_candidate=False))
                continue
            candidate = dict(id=record_id, prompt=prompt, source=name, category='coursework', group=group_key(prompt))
            if name == 'train_sft.json':
                candidate['answer'] = item.get('output', '')
                answers = [candidate['answer']]
            else:
                candidate.update(chosen=item.get('chosen', ''), rejected=item.get('rejected', ''))
                answers = [candidate['chosen'], candidate['rejected']]
            if any(not isinstance(a, str) or not a.strip() for a in answers):
                reason = 'missing_completion'
            if len(answers) == 2 and answers[0] == answers[1]:
                reason = 'identical_pair'
            words = set(prompt.casefold().split())
            if any(words and len(words & ts) / len(words | ts) >= .8 for ts in tokens):
                reason = 'evaluation_overlap_candidate'
            review = reviews.get(record_id, {})
            if not reason and not (review.get('approved') is True and review.get('reason', '').strip()) and not (name=='train_sft.json' and config.get('structural_sft_control',False)):
                reason = 'needs_content_review'
            audit.append(dict(id=record_id, file=name, index=index, reason=reason, meta_instruction_candidate=bool(re.search(r'(preference pair|chosen.*rejected|generate.*pair)', prompt, re.I))))
            if not reason:
                target.append(candidate)
    if config.get('synthetic', True):
        for row in synthetic_rows():
            row['id'] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            # Programmatic oracles are approved, but still subject to overlap screening.
            words=set(row['prompt'].casefold().split())
            if any(words and len(words & ts)/len(words | ts)>=.8 for ts in tokens):
                continue
            rows.append(row)
            if row['category']=='math':
                wrong='The answer is: '+str(int(row['answer'].rsplit(' ',1)[-1])+1)
            elif row['category']=='logic':
                wrong='no' if row['answer']=='yes' else 'yes'
            else:
                wrong=row['answer'].replace('    return x ', '    return 1 + x ')
            pairs.append(dict(id=row['id']+'-pair',prompt=row['prompt'],chosen=row['answer'],rejected=wrong,group=row['group'],source=row['source'],reason='Known incorrect result constructed from the same arithmetic or entailment oracle'))
    if config.get('gsm_train'):
        for index,line in enumerate(Path(config['gsm_train']).read_text(encoding='utf-8').splitlines()):
            item=json.loads(line)
            prompt=item['question']; answer=item['answer']
            words=set(prompt.casefold().split())
            if any(words and len(words & ts)/len(words | ts)>=.8 for ts in tokens):
                continue
            if '####' not in answer: raise ValueError('GSM training record lacks final answer')
            reasoning,final=answer.rsplit('####',1)
            row=dict(id=f'gsm-train-{index}',prompt=prompt,answer=reasoning.strip()+'\nThe answer is: '+final.strip(),category='math',group=group_key(prompt),source='GSM8K train')
            rows.append(row)
    unique = {}
    for row in rows:
        previous = unique.get(row['prompt'])
        if previous and previous['answer'] != row['answer']:
            raise ValueError('Conflicting completions for duplicate prompt')
        unique[row['prompt']] = row
    rows = list(unique.values())
    # A prompt present in multiple sources must not acquire different split groups.
    parents = {}
    prompt_groups = {}
    def canonical(group):
        while group in parents:
            group = parents[group]
        return group
    for row in rows + pairs:
        normalized = group_key(row['prompt'])
        current = canonical(row['group'])
        if normalized in prompt_groups:
            previous = canonical(prompt_groups[normalized])
            if current != previous:
                parents[current] = previous
        prompt_groups[normalized] = canonical(current)
    for row in rows + pairs:
        row['group'] = canonical(row['group'])
    def partition(items):
        train, validation = [], []
        for row in items:
            bucket = int(hashlib.sha256(row['group'].encode()).hexdigest()[:8], 16) % 5
            (validation if bucket == 0 else train).append(row)
        return train, validation
    sft_train, sft_val = partition(rows)
    dpo_train, dpo_val = partition(pairs)
    if not sft_train or not sft_val:
        raise ValueError('Need nonempty train and validation groups')
    files = dict(sft_train=sft_train, sft_validation=sft_val, dpo_train=dpo_train, dpo_validation=dpo_val)
    for name, data in files.items():
        save(out / (name + '.json'), data)
    save(out / 'audit.json', audit)
    save(out / 'manifest.json', dict(counts={k:len(v) for k,v in files.items()}, quarantined=sum(a['reason'] is not None for a in audit), hashes={name:sha(out / (name+'.json')) for name in files}, sources={name:sha(raw/name) for name in ['train_sft.json','train_dpo.json']}))


def sequence_logps(model, example, device):
    ids = torch.tensor([example['input_ids']], device=device)
    labels = torch.tensor([example['labels']], device=device)[:, 1:]
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[:, :-1].float()
    values = F.log_softmax(logits, dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return (values * labels.ne(-100)).sum(-1)


def preference_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    return -F.logsigmoid(beta * ((chosen - rejected) - (ref_chosen - ref_rejected))).mean()


def train(config, mode):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    random.seed(config.get('seed', 42)); torch.manual_seed(config.get('seed', 42))
    if config.get('steps',200)<1 or config.get('accumulation',8)<1:
        raise ValueError('Positive training steps and accumulation are required')
    if mode=='dpo' and config.get('overfit'):
        raise ValueError('Overfit diagnostic is an SFT operation')
    torch.set_num_threads(1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    out = Path(config['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise ValueError('Training output exists; choose a new directory')
    if not config.get('overfit'):
        if not config.get('overfit_report') or not read(config['overfit_report'])['passed']:
            raise ValueError('Pass the reloaded 16-example overfit check before full training')
        if not config.get('dev_suite_dir'):
            raise ValueError('Full training requires an independent development suite')
    if mode=='dpo' and not config.get('sft_improved_on_dev',False):
        raise ValueError('Verify SFT improvement on development tasks before DPO')
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    tokenizer.chat_template = TEMPLATE
    if config.get('eos_token'): tokenizer.eos_token = config['eos_token']
    model = AutoModelForCausalLM.from_pretrained(config['model'], dtype=dtype, local_files_only=True).to(device)
    if mode == 'dpo' or config.get('initial_adapter'):
        if not (config.get('sft_adapter') or config.get('initial_adapter')):
            raise ValueError('DPO requires a saved SFT adapter')
        model = PeftModel.from_pretrained(model, config.get('initial_adapter',config.get('sft_adapter')), is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=config.get('rank',16), lora_alpha=32, lora_dropout=0, target_modules=['q_proj','k_proj','v_proj','o_proj'], task_type='CAUSAL_LM'))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    data = read(Path(config['data_dir']) / ('dpo_train.json' if mode == 'dpo' else 'sft_train.json'))
    if mode == 'dpo' and not read(Path(config['data_dir']) / 'dpo_validation.json'):
        raise ValueError('DPO needs independently reviewed validation pairs')
    if config.get('overfit'):
        # Short, distinct completions from training only.
        seen, selected = set(), []
        for row in data:
            if row['answer'] not in seen and row['category'] == 'math':
                selected.append(row); seen.add(row['answer'])
            if len(selected) == 16: break
        if len(selected) != 16: raise ValueError('Need 16 distinct short training answers')
        data = selected
    examples, rejected_count = [], 0
    for row in data:
        try:
            if mode == 'dpo':
                example = [encode(tokenizer,row['prompt'],row[k],config.get('max_length',512)) for k in ['chosen','rejected']]
            else:
                example = encode(tokenizer,row['prompt'],row['answer'],config.get('max_length',512))
            examples.append((row, example))
        except ValueError:
            rejected_count += 1
    if not examples: raise ValueError('No complete examples fit the sequence budget')
    if mode=='dpo' and len(examples)<config.get('accumulation',8):
        raise ValueError('DPO needs at least one complete accumulation batch')
    if mode=='sft' and not config.get('overfit'):
        random.shuffle(examples)
    reference = []
    validation_pairs = []
    validation_reference = []
    if mode == 'dpo':
        model.eval()
        for row in read(Path(config['data_dir'])/'dpo_validation.json'):
            try:
                validation_pairs.append([encode(tokenizer,row['prompt'],row[k],config.get('max_length',512)) for k in ['chosen','rejected']])
            except ValueError:
                continue
        if not validation_pairs:raise ValueError('No complete validation preference pairs fit')
        with torch.no_grad():
            reference = [(sequence_logps(model,a,device).item(),sequence_logps(model,b,device).item()) for _,(a,b) in examples]
            validation_reference = [(sequence_logps(model,a,device).item(),sequence_logps(model,b,device).item()) for a,b in validation_pairs]
        cap_path = out/'reference-cache.json'
        save(cap_path,dict(values=reference,adapter_hash=sha(Path(config['sft_adapter'])/'adapter_model.safetensors'),examples_hash=hashlib.sha256(json.dumps(examples,sort_keys=True).encode()).hexdigest(),template=TEMPLATE,eos_token_id=tokenizer.eos_token_id,config=config))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.get('learning_rate', 5e-6 if mode == 'dpo' else 1e-4))
    accumulation = config.get('accumulation',8)
    steps = config.get('steps',200)
    if mode=='dpo': steps=min(steps,max(1,len(examples)//accumulation))
    history = []
    best = -1.0
    supervised_tokens = 0
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        total = 0.0
        for micro in range(accumulation):
            index = (step*accumulation+micro) % len(examples)
            _, example = examples[index]
            completions = example if mode=='dpo' else [example]
            supervised_tokens += sum(sum(value!=-100 for value in part['labels']) for part in completions)
            if mode == 'dpo':
                a,b = example
                loss = preference_loss(sequence_logps(model,a,device),sequence_logps(model,b,device),*reference[index],config.get('beta',.1))
            else:
                inputs = {k:torch.tensor([v],device=device) for k,v in example.items()}
                loss = model(**inputs, attention_mask=torch.ones_like(inputs['input_ids'])).loss
            if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
            (loss/accumulation).backward(); total += loss.item()/accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        history.append(dict(step=step+1,loss=total))
        if (step+1) % 10 == 0: print(history[-1],flush=True)
        if not config.get('overfit') and ((step+1)%50==0 or step+1==steps):
            model.save_pretrained(out/'last');tokenizer.save_pretrained(out/'last')
            save(out/'progress.json',dict(step=step+1,history=history,config=config))
            result = evaluate_dev(model,tokenizer,config['dev_suite_dir'],device)
            save(out/f'dev-{step+1}.json',result)
            if result['objective_mean'] > best:
                best = result['objective_mean']
                model.save_pretrained(out/'best');tokenizer.save_pretrained(out/'best')
                save(out/'selection.json',dict(step=step+1,objective_mean=best))
            model.train()
    model.save_pretrained(out); tokenizer.save_pretrained(out)
    if mode=='dpo':
        model.eval()
        with torch.no_grad():
            margins=[config.get('beta',.1)*(sequence_logps(model,a,device).item()-sequence_logps(model,b,device).item()-ra+rb) for (a,b),(ra,rb) in zip(validation_pairs,validation_reference)]
        save(out/'preference-validation.json',dict(n=len(margins),reward_accuracy=sum(m>0 for m in margins)/len(margins),mean_reward_margin=sum(margins)/len(margins),note='Held-out synthetic oracle preferences; not a general human preference benchmark'))
    save(out/'training-report.json',dict(config=config,examples=len(examples),supervised_tokens=supervised_tokens,excluded_long_or_invalid=rejected_count,history=history,dtype=str(dtype),eos_token_id=tokenizer.eos_token_id,versions={name:importlib.metadata.version(name) for name in ['torch','transformers','peft']},peak_memory_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,training_data_hash=sha(Path(config['data_dir'])/('dpo_train.json' if mode=='dpo' else 'sft_train.json'))))
    if config.get('overfit'):
        # Reload from disk to catch saving/loading mistakes.
        del model, optimizer
        if device == 'cuda': torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(config['model'],dtype=dtype,local_files_only=True).to(device)
        model = PeftModel.from_pretrained(base,str(out)).eval()
        checks = []
        with torch.inference_mode():
            for row,_ in examples:
                inputs = tokenizer(prompt_text(row['prompt']),return_tensors='pt').to(device)
                generated = model.generate(**inputs,do_sample=False,max_new_tokens=32,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id)
                response = tokenizer.decode(generated[0,inputs.input_ids.shape[1]:],skip_special_tokens=True).strip()
                checks.append(dict(id=row['id'],expected=row['answer'],response=response,correct=response==row['answer']))
        save(out/'overfit-check.json',dict(passed=sum(r['correct'] for r in checks)>=14,correct=sum(r['correct'] for r in checks),total=len(checks),records=checks))


def evaluate_dev(model, tokenizer, suite_dir, device):
    from .capability import score_task
    model.eval()
    rows=[]
    eos_ids=list({tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('<|im_end|>')})
    with torch.inference_mode():
        for task in read(Path(suite_dir)/'suite.json')['tasks']:
            inputs=tokenizer(prompt_text(task['prompt']),return_tensors='pt').to(device)
            output=model.generate(**inputs,do_sample=False,max_new_tokens=256,eos_token_id=eos_ids,pad_token_id=tokenizer.eos_token_id,use_cache=True)
            generated=output[0,inputs.input_ids.shape[1]:]
            row=dict(stage='candidate',response=tokenizer.decode(generated,skip_special_tokens=True),finish_reason='eos' if generated[-1].item() in eos_ids else 'length')
            rows.append(dict(response=row['response'],**score_task(task,row,execute_code=True)))
    means=[]
    for category in ['math','logic','code']:
        subset=[r for r in rows if r['category']==category]
        if not subset or any(r['correct'] is None for r in subset):
            raise ValueError('Development selection requires all objective tasks scored')
        means.append(sum(r['correct'] for r in subset)/len(subset))
    return dict(objective_mean=sum(means)/3,records=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','sft','dpo','paired-freeze','paired-collect','paired-run',
        'progression-prepare','progression-run','progression-public','progression-report'])
    parser.add_argument('--config',required=True,type=Path)
    args = parser.parse_args()
    config = read(args.config)
    if args.command.startswith('progression-'):
        from .progression import dispatch
        return dispatch(args.command.removeprefix('progression-'),config)
    if args.command.startswith('paired-'):
        from .paired_experiment import collect, freeze
        root=Path(config['output_dir'])
        suite=freeze(config,root)
        if args.command=='paired-freeze': return
        pairs=collect(config,root,suite)
        if args.command=='paired-collect': return
        from .paired_experiment import run_comparison
        return run_comparison(config,root,suite,pairs)
    if args.command == 'prepare': prepare(config)
    else: train(config,args.command)


if __name__ == '__main__':
    main()
