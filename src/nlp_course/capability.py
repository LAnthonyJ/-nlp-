"""Resumable real-checkpoint evaluation, isolated code scoring and blind review.

Default workspace artifacts live in ignored outputs/. No fabricated scores.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import time
import uuid
import threading
from types import SimpleNamespace

from .capability_data import prepare, digest
from .evaluation import extract_final_answer

STAGES = ('base','sft','dpo')
MODEL = 'Qwen/Qwen2.5-0.5B'


def save(path, data):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def records(path):
    if not Path(path).exists():return []
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def generate(suite_dir, run_dir, stage, train_dir, phase='formal', open_only=False):
    import torch
    torch.set_num_threads(1)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    if not torch.cuda.is_available():raise ValueError('CUDA evaluation requires the separate GPU environment')
    model_path=snapshot_download(MODEL,local_files_only=True)
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    tokenizer.chat_template="{% for m in messages %}<|im_start|>{{m['role']}}\n{{m['content']}}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    model=AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.float16,local_files_only=True).to('cuda')
    weights={'base':sha(Path(model_path)/'model.safetensors')}
    weights['model_config']=sha(Path(model_path)/'config.json')
    weights['tokenizer_config']=sha(Path(model_path)/'tokenizer_config.json')
    if stage!='base':
        from peft import PeftModel
        adapter=Path(train_dir)/f'q2_{stage}_output'
        weights['adapter']=sha(adapter/'adapter_model.safetensors')
        weights['adapter_config']=sha(adapter/'adapter_config.json')
        model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False)
    model.eval()
    suite=read(Path(suite_dir)/'suite.json')
    if suite['near_overlap_candidates']:raise ValueError('Review training-overlap candidates before evaluating models')
    tasks=suite['tasks']
    if phase=='formal':
        smoke=records(Path(run_dir)/'smoke'/f'{stage}.jsonl')
        if len(smoke)!=10 or any(not r['response'].strip() for r in smoke):raise ValueError('Complete the ten development prompts before formal generation')
    else:
        from .decoding import DecodingSampler
        messages=[dict(role='user',content='What is 2+3?')]
        encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt',return_dict=True).to('cuda')
        with torch.inference_mode():
            probe=model.generate(**encoded,do_sample=False,max_new_tokens=8,pad_token_id=tokenizer.eos_token_id)
        expected=tokenizer.decode(probe[0,encoded.input_ids.shape[1]:],skip_special_tokens=True).strip()
        actual=DecodingSampler(tokenizer.convert_tokens_to_ids('<|im_end|>')).generate(model,tokenizer,messages,max_new_tokens=8)
        if actual!=expected:raise ValueError('Custom and reference greedy decoder disagree on the development probe')
        save(Path(run_dir)/'smoke'/f'{stage}-decoder-check.json',dict(expected=expected,actual=actual,passed=True))
    if phase=='smoke':
        tasks=[dict(id=f'smoke-{i}',category='logic',split='smoke',prompt=prompt,reference='development only') for i,prompt in enumerate([
            'What is 2+3? End with The answer is: NUMBER.', 'What is 6*7? End with The answer is: NUMBER.',
            'A is before B. Return their order as A,B.', 'All cats are mammals. Is a cat a mammal? Answer yes or no.',
            'Return a Python function double(x) that returns x*2.', 'Return a Python function empty() that returns an empty list.',
            'Explain what an atom is in one sentence.', 'Explain the difference between weather and climate.',
            'Rewrite politely: Send me the file.', 'Write exactly two sentences about planting a tree.'])]
    metadata=dict(stage=stage,phase=phase,model=MODEL,weights=weights,suite_hash=digest(suite),
                  seed=42,dtype='float16',max_new_tokens=256,decoder='greedy',torch=torch.__version__,protocol_version=2,
                  math_suffix='At the end, write The answer is: followed by your final numeric answer.')
    folder=Path(run_dir)/phase; folder.mkdir(parents=True,exist_ok=True)
    config=folder/f'{stage}-config.json'
    if config.exists() and read(config)!=metadata:raise ValueError('Resume configuration changed; use a new run directory')
    save(config,metadata)
    path=folder/f'{stage}.jsonl'; old=records(path)
    done={r['task_id'] for r in old}
    if len(done)!=len(old):raise ValueError('Duplicate task IDs in existing output')
    if any(r['task_id'] not in {t['id'] for t in tasks} or r['stage']!=stage for r in old):raise ValueError('Resume records do not match this stage/suite')
    eos=tokenizer.convert_tokens_to_ids('<|im_end|>')
    eos_ids=list(set([eos,tokenizer.eos_token_id]))
    for task in tasks:
        if open_only and not (task['split']=='main' and task['category'] in ('concept','writing')):continue
        if task['id'] in done:continue
        prompt=task['prompt']
        if task['category']=='math':prompt+='\n'+metadata['math_suffix']
        text='<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n'
        encoded=tokenizer(text,return_tensors='pt').to('cuda')
        budget=task.get('max_new_tokens',256)
        if encoded.input_ids.shape[1]+budget>model.config.max_position_embeddings:raise ValueError('Context limit exceeded')
        torch.manual_seed(42); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        with torch.inference_mode():
            out=model.generate(**encoded,do_sample=False,max_new_tokens=budget,eos_token_id=eos_ids,pad_token_id=tokenizer.eos_token_id,use_cache=True)
        torch.cuda.synchronize(); generated=out[0,encoded.input_ids.shape[1]:]
        response=tokenizer.decode(generated,skip_special_tokens=True)
        row=dict(task_id=task['id'],stage=stage,response=response,input_tokens=encoded.input_ids.shape[1],
                 output_tokens=len(generated),seconds=time.perf_counter()-started,
                 finish_reason='eos' if len(generated) and generated[-1].item() in eos_ids else 'length',
                 peak_memory_bytes=torch.cuda.max_memory_allocated())
        with path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
        print(f'{stage}: {task["id"]} ({len(generated)} tokens)',flush=True)


def _run_limited(command, text, timeout=15, max_output=131072, cleanup=None):
    """Bound child output in memory; production callers must supply a sandbox."""
    payload=text.encode('utf-8')
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    buffers=[bytearray(),bytearray()];counts=[0,0];exceeded=threading.Event()
    def drain(stream,index):
        while True:
            chunk=stream.read(4096)
            if not chunk:break
            counts[index]+=len(chunk);buffers[index].extend(chunk)
            if len(buffers[index])>65536:del buffers[index][:-65536]
            if sum(counts)>max_output:exceeded.set()
    threads=[threading.Thread(target=drain,args=(stream,index),daemon=True) for index,stream in enumerate([process.stdout,process.stderr])]
    for thread in threads:thread.start()
    def feed():
        try:process.stdin.write(payload)
        except OSError:pass
        finally:
            try:process.stdin.close()
            except OSError:pass
    writer=threading.Thread(target=feed,daemon=True);writer.start()
    started=time.monotonic();timed_out=False
    while process.poll() is None:
        if exceeded.is_set() or time.monotonic()-started>timeout:
            timed_out=not exceeded.is_set()
            process.kill()
            if cleanup is not None:cleanup()
            break
        time.sleep(.02)
    process.wait(timeout=5)
    writer.join(timeout=2)
    for thread in threads:thread.join(timeout=2)
    for stream in [process.stdout,process.stderr]:stream.close()
    return SimpleNamespace(returncode=process.returncode,stdout=buffers[0].decode('utf-8','replace'),stderr=buffers[1].decode('utf-8','replace'),timed_out=timed_out,output_limit=exceeded.is_set())


def docker_score(task, response, image='python:3.11-slim'):
    """Generated code never executes on the host; no mounts or network."""
    blocks=re.findall(r'```(?:python)?\s*\n(.*?)```',response,re.S)
    code=blocks[0] if len(blocks)==1 else response.strip()
    if 'def '+task['entry_point'] not in code:return dict(status='missing_function',correct=False,detail='Missing requested function definition')
    try:ast.parse(code)
    except SyntaxError as exc:return dict(status='syntax_error',correct=False,detail=str(exc))
    # Preserve the imports supplied with the original HumanEval task, so e.g.
    # a correct function using List annotations is not failed by a missing harness import.
    original=task.get('prompt','').split('\n',1)[-1]
    try:
        parsed=ast.parse(original)
        imports='\n'.join(ast.get_source_segment(original,node) for node in parsed.body if isinstance(node,(ast.Import,ast.ImportFrom)))
    except SyntaxError:imports=''
    marker='TESTS_COMPLETED_'+uuid.uuid4().hex
    script=imports+'\n'+code+'\n'+task['tests']+'\ncheck('+task['entry_point']+')\nprint('+repr('\n'+marker)+')\n'
    name='nlp-eval-'+uuid.uuid4().hex[:16]
    args=['docker','run','--name',name,'--rm','--network','none','--read-only','--memory','256m','--cpus','1',
          '--pids-limit','32','--cap-drop','ALL','--security-opt','no-new-privileges','--user','65534:65534',
          '--tmpfs','/tmp:rw,noexec,nosuid,size=16m', '-i',image,'python','-I','-']
    def cleanup():subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
    run=_run_limited(args,script,cleanup=cleanup)
    if run.timed_out:
        return dict(status='timeout',correct=False)
    if run.output_limit:return dict(status='output_limit',correct=False)
    if run.returncode in (125,126,127):return dict(status='unavailable',correct=None)
    passed=run.returncode==0 and marker in run.stdout.splitlines()
    return dict(status='correct' if passed else 'test_failure',correct=passed,returncode=run.returncode,stderr_tail=run.stderr[-4000:] if not passed else '')


def _bounded_number(text):
    from .evaluation import parse_number
    if not isinstance(text,str) or len(text)>4096:return None
    for exponent in re.findall(r'[eE]([+-]?\d+)',text):
        if len(exponent.lstrip('+-').lstrip('0'))>5 or abs(int(exponent))>10000:return None
    return parse_number(text)


def _relaxed_numeric_answer(response):
    # Preserve last-answer and nested-box semantics before relaxing numeric syntax.
    normalized=re.sub(r'the (?:final )?answer is\s*:?', 'The answer is:',response,flags=re.I)
    answer=extract_final_answer(normalized)
    exact=_bounded_number(answer)
    if exact is not None:return exact
    if answer is None:return None
    match=re.match(r'^\$?([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?:/\d+)?)(?![\w,/]|\.\d)',answer)
    if not match:return None
    tail=answer[match.end():].lstrip()
    if tail and tail[0] in '+-*/^=<>[{\\':return None
    return _bounded_number(match[1])


def score_task(task, row, execute_code=False):
    response=row['response']; category=task['category']; score=dict(correct=None,status='pending_human')
    if category=='math':
        predicted=_bounded_number(extract_final_answer(response))
        expected=_bounded_number(task['reference'])
        if expected is None:raise ValueError('Math reference must be a supported numeric answer')
        correct=predicted is not None and predicted==expected
        score=dict(correct=correct,status='invalid_answer' if predicted is None else 'correct' if correct else 'incorrect')
        # Secondary format-sensitivity diagnostic: never replace the strict score.
        # Requires an explicit final-answer phrase, not an arbitrary last number.
        relaxed=_relaxed_numeric_answer(response)
        score['format_relaxed_correct']=relaxed is not None and relaxed==expected
        score['format_relaxed_extracted']=str(relaxed) if relaxed is not None else None
    elif category=='logic':
        candidate=response.strip().rstrip('.')
        candidate=re.sub(r'\s+','',candidate)
        correct=candidate.casefold()==task['reference'].casefold()
        score=dict(correct=correct,status='correct' if correct else 'incorrect_or_format')
    elif category=='code':
        score=docker_score(task,response) if execute_code else dict(correct=None,status='unavailable')
    elif task['split']=='diagnostic':
        if category=='concept':
            score=dict(correct=None,status='lexical_probe',keyword_coverage=sum(k in response.casefold() for k in task['keywords'])/len(task['keywords']))
        else:
            lines=[s for s in response.splitlines() if s.strip()]
            passed=len(lines)==3
            if task['condition']=='three':passed=passed and all(s.startswith('- ') for s in lines) and response.rstrip().endswith('Reuse matters.')
            score=dict(correct=passed,status='correct' if passed else 'constraint_failure')
    if row['finish_reason']=='length':score['truncated']=True
    return dict(task_id=task['id'],stage=row['stage'],category=category,split=task['split'],**score)


def score(suite_dir,run_dir,execute_code=False,progress=False):
    tasks=read(Path(suite_dir)/'suite.json')['tasks']; by_id={t['id']:t for t in tasks}
    results=[]
    if execute_code:
        check=subprocess.run(['docker','info','--format','{{.ServerVersion}}'],capture_output=True,timeout=20)
        if check.returncode:raise ValueError('Docker engine unavailable; no generated code executed')
        image=subprocess.run(['docker','image','inspect','python:3.11-slim'],capture_output=True,timeout=20)
        if image.returncode:raise ValueError('Prepare official python:3.11-slim image before code scoring')
    for stage in STAGES:
        rows=records(Path(run_dir)/'formal'/f'{stage}.jsonl')
        ids=[r['task_id'] for r in rows]
        if len(ids)!=len(set(ids)) or any(i not in by_id for i in ids):raise ValueError('Unknown/duplicate output task')
        for r in rows:
            results.append(score_task(by_id[r['task_id']],r,execute_code))
            if progress and len(results)%10==0:print(f'Scored {len(results)} responses',flush=True)
    save(Path(run_dir)/'scores.json',dict(expected_per_stage=len(tasks),records=results,
         code_execution='isolated Docker python:3.11-slim' if execute_code else 'not executed'))
    return results


def blind_review(suite_dir,run_dir):
    tasks=read(Path(suite_dir)/'suite.json')['tasks']; wanted={t['id']:t for t in tasks if t['split']=='main' and t['category'] in ('concept','writing')}
    rows=[]
    for stage in STAGES:
        for row in records(Path(run_dir)/'formal'/f'{stage}.jsonl'):
            if row['task_id'] in wanted:rows.append((wanted[row['task_id']],row))
    if len(rows)!=60:raise ValueError('Need all 60 open-task responses before exporting blind review')
    rng=random.Random(42);rng.shuffle(rows)
    rows.extend(rng.sample(rows,6));rng.shuffle(rows)
    mapping={}; seen=set();path=Path(run_dir)/'blind-review.csv'
    if path.exists():raise ValueError('Blind review sheet already exists; refusing to overwrite human ratings')
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['review_id','category','prompt','reference_rubric','response','accuracy_or_completion','completeness_or_coherence','clarity_or_expression','notes'])
        writer.writeheader()
        for i,(task,row) in enumerate(rows):
            key=f'R{i:03d}';pair=(task['id'],row['stage'])
            mapping[key]=dict(task_id=task['id'],stage=row['stage'],is_repeat=pair in seen)
            seen.add(pair)
            writer.writerow(dict(review_id=key,category=task['category'],prompt=task['prompt'],reference_rubric=task['reference'],response=row['response']))
    save(Path(run_dir)/'private-review-key.json',mapping)


def import_review(run_dir,path):
    key=read(Path(run_dir)/'private-review-key.json'); received={}
    fields=['accuracy_or_completion','completeness_or_coherence','clarity_or_expression']
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            rid=row['review_id']
            if rid not in key or rid in received:raise ValueError('Unknown/duplicate review ID')
            values=[int(row[f]) for f in fields]
            if any(v<1 or v>5 for v in values):raise ValueError('Human ratings must be integers 1 through 5')
            received[rid]=dict(**key[rid],ratings=values,notes=row.get('notes',''))
    if set(received)!=set(key):raise ValueError('Human review is incomplete')
    save(Path(run_dir)/'human-scores.json',list(received.values()))


def wilson(correct,total):
    if not total:return None
    z=1.959963984540054;p=correct/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [0.0 if correct==0 else max(0,center-half),1.0 if correct==total else min(1,center+half)]


def analyze(suite_dir,run_dir):
    suite=read(Path(suite_dir)/'suite.json'); scored=read(Path(run_dir)/'scores.json')['records']
    results=[]
    for stage in STAGES:
        for category in ['math','logic','code','concept','writing']:
            rows=[r for r in scored if r['stage']==stage and r['split']=='main' and r['category']==category]
            measured=[r for r in rows if r['correct'] is not None]
            correct=sum(r['correct'] for r in measured)
            results.append(dict(stage=stage,category=category,expected=sum(t['category']==category and t['split']=='main' for t in suite['tasks']),received=len(rows),scored=len(measured),correct=correct,accuracy=correct/len(measured) if measured else None,interval=wilson(correct,len(measured)),statuses=dict(Counter(r['status'] for r in rows))))
            if category=='math':
                results[-1]['format_relaxed_correct']=sum(r.get('format_relaxed_correct',False) for r in rows)
                results[-1]['format_relaxed_accuracy']=results[-1]['format_relaxed_correct']/len(rows) if rows else None
                results[-1]['format_relaxed_interval']=wilson(results[-1]['format_relaxed_correct'],len(rows))
    human=read(Path(run_dir)/'human-scores.json') if (Path(run_dir)/'human-scores.json').exists() else None
    pairs=[]
    for stage in STAGES:
        for category in ['math','logic','code','concept','writing']:
            group=[t for t in suite['tasks'] if t['category']==category and t['split']=='diagnostic']
            for pair in sorted(set(t['pair'] for t in group)):
                ids={t['id']:t['condition'] for t in group if t['pair']==pair}
                pairs.append(dict(stage=stage,category=category,pair=pair,conditions=[dict(condition=ids[r['task_id']],**r) for r in scored if r['stage']==stage and r['task_id'] in ids]))
    human_summary=[]
    for stage in STAGES:
        for category in ['concept','writing']:
            selected=[r for r in human or [] if not r.get('is_repeat') and r['stage']==stage and r['task_id'].startswith(category+'-')]
            human_summary.append(dict(stage=stage,category=category,n=len(selected),means=[sum(r['ratings'][i] for r in selected)/len(selected) for i in range(3)] if selected else None))
    repetition=[]
    for r in human or []:
        if r.get('is_repeat'):
            original=next(x for x in human if not x.get('is_repeat') and x['task_id']==r['task_id'] and x['stage']==r['stage'])
            repetition.append(sum(abs(a-b) for a,b in zip(r['ratings'],original['ratings']))/3)
    changes=[]
    lookup={(r['stage'],r['task_id']):r for r in scored}
    for first,last in [('base','sft'),('sft','dpo')]:
        for category in ['math','logic','code']:
            counts=Counter()
            for t in suite['tasks']:
                a,b=lookup.get((first,t['id'])),lookup.get((last,t['id']))
                if t['split']!='main' or t['category']!=category or not a or not b or a['correct'] is None or b['correct'] is None:continue
                counts['improved' if b['correct'] and not a['correct'] else 'regressed' if a['correct'] and not b['correct'] else 'unchanged']+=1
            changes.append(dict(first=first,last=last,category=category,counts=dict(counts)))
    logic_refs=Counter(t['reference'] for t in suite['tasks'] if t['category']=='logic' and t['split']=='main')
    logic_baseline=dict(answer=logic_refs.most_common(1)[0][0],accuracy=logic_refs.most_common(1)[0][1]/sum(logic_refs.values())) if logic_refs else None
    report=dict(summary=results,boundaries=pairs,stage_changes=changes,human_summary=human_summary,logic_constant_answer_baseline=logic_baseline,
                human_repeat_mean_difference=sum(repetition)/len(repetition) if repetition else None,
                human_status='scored' if human else 'pending',human_records=human,
                limitations=['English quick diagnostic, not a comprehensive benchmark','Generated logic templates are simple and their answer distribution is imbalanced; inspect the constant-answer baseline','Single training run; stage differences do not establish general causality','Base uses the same instruction protocol as SFT/DPO; a gap may reflect instruction alignment as well as task skill','Lexical overlap screening does not establish semantic independence or absence of pretraining contamination','Keyword coverage is a lexical probe, not factual accuracy','Missing/unsafe code execution is not counted as model failure'])
    save(Path(run_dir)/'analysis.json',report)
    lines=['# Model capability diagnostic','',f'Human review: {report["human_status"]}','', '| Stage | Task | Received/expected | Scored | Correct | Accuracy |','|---|---|---:|---:|---:|---:|']
    for r in results:
        value=f'{r["accuracy"]:.1%}' if r['accuracy'] is not None else 'Pending/unavailable'
        lines.append(f'| {r["stage"]} | {r["category"]} | {r["received"]}/{r["expected"]} | {r["scored"]} | {r["correct"]} | {value} |')
    lines.extend(['','## Limits',*['- '+s for s in report['limitations']]])
    lines.extend(['','## Math format sensitivity (secondary, post-hoc)','',
                  'Strict scores include the requested answer protocol. The following relaxation diagnoses format sensitivity; it does not replace the strict result.'])
    for r in results:
        if r['category']=='math':lines.append(f'- {r["stage"]}: strict {r["correct"]}/{r["scored"]}; relaxed final-answer format {r["format_relaxed_correct"]}/{r["received"]}.')
    lines.extend(['','## Failure evidence / 失败证据','', 'Only observable errors and controlled interventions are evidence. Capacity and training-data explanations remain hypotheses.'])
    selected=[]
    for stage in STAGES:
        candidates=[]
        for category in ['math','logic','code']:
            failures=sorted([r for r in scored if r['stage']==stage and r['category']==category and r['split']=='main' and r['correct'] is False],key=lambda r:r['task_id'])
            candidates.extend(failures[:1])
        originals={r['task_id']:r for r in records(Path(run_dir)/'formal'/f'{stage}.jsonl')}
        for r in candidates:
            task=next(t for t in suite['tasks'] if t['id']==r['task_id'])
            selected.append(dict(task=task,response=originals[r['task_id']],score=r))
            lines.extend(['',f'### {stage}: {r["task_id"]}',task['prompt'],'',originals[r['task_id']]['response'],'',f'Observed status: {r["status"]}.'])
    save(Path(run_dir)/'cases.json',selected)
    transitions=[]
    for first,last in [('base','sft'),('sft','dpo')]:
        for direction in ['improved','regressed']:
            candidates=[]
            for task in sorted(suite['tasks'],key=lambda t:t['id']):
                a,b=lookup.get((first,task['id'])),lookup.get((last,task['id']))
                if task['split']!='main' or not a or not b or a['correct'] is None or b['correct'] is None:continue
                if (direction=='improved' and not a['correct'] and b['correct']) or (direction=='regressed' and a['correct'] and not b['correct']):candidates.append(task['id'])
            transitions.append(dict(first=first,last=last,direction=direction,task_ids=candidates[:3]))
    save(Path(run_dir)/'transition-cases.json',transitions)
    (Path(run_dir)/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    zh=['# 真实模型能力快速诊断','',f'开放题人工评分状态：{"已导入" if human else "待你填写匿名评分表"}。',
        '', '| 模型阶段 | 任务 | 已收到/应收到 | 已评分 | 正确 | 通过率 |','|---|---|---:|---:|---:|---:|']
    for r in results:
        value=f'{r["accuracy"]:.1%}' if r['accuracy'] is not None else '待评分或环境不可用'
        zh.append(f'| {r["stage"]} | {r["category"]} | {r["received"]}/{r["expected"]} | {r["scored"]} | {r["correct"]} | {value} |')
    zh.extend(['','## 数学答案格式敏感性（事后补充分析）','',
        '严格分数包含答案输出协议。以下放宽提取仅诊断格式敏感性，不替代严格分数，也不证明推理过程正确。'])
    for r in results:
        if r['category']=='math':zh.append(f'- {r["stage"]}：严格 {r["correct"]}/{r["scored"]}；放宽最终答案格式 {r["format_relaxed_correct"]}/{r["received"]}。')
    zh.extend(['','## 能力边界与解释','',
        '- 数字题的答案错误与答案提取失败分别统计；格式失败不能直接解释为不会计算。',
        '- 代码分数只统计已完成隔离执行的回答；缺失环境不计为模型错误。',
        '- 受控题对只改变一个条件，用于定位数字规模、推理长度、上下文、约束和输出预算的影响。',
        '- 概念诊断中的关键词覆盖率不是事实正确率；开放题质量等待匿名人工评分。',
        '- 三个阶段各只有一次训练权重；阶段差异不能证明训练方法的一般因果效果。',
        '- 模型容量、训练数据质量和偏好方向属于待验证解释，须结合审计样本与具体失败证据。',
        '- 本轮仅为英文小样本快速诊断；中文、多轮对话、长上下文与完整基准均未覆盖。',
        '', '具体失败回答见 cases.json，逐题对照与受控题对见 analysis.json。'])
    (Path(run_dir)/'report.zh-CN.md').write_text('\n'.join(zh)+'\n',encoding='utf-8')
    return report


def render_report(run_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=read(Path(run_dir)/'analysis.json')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,category in zip(axes,['math','logic','code']):
        for index,stage in enumerate(STAGES):
            row=next(r for r in report['summary'] if r['stage']==stage and r['category']==category)
            if row['accuracy'] is None:ax.text(index,.05,'Unavailable',ha='center',rotation=90)
            else:
                low,high=row['interval'];p=row['accuracy']
                ax.bar(index,p,color=['#64748b','#0891b2','#2563eb'][index])
                ax.errorbar(index,p,yerr=[[p-low],[high-p]],fmt='o',color='black',capsize=4,markersize=4)
                ax.annotate(f'{row["correct"]}/{row["scored"]}',(index,p),xytext=(-7,9),textcoords='offset points',ha='right',fontsize=8)
                if category=='math':
                    relaxed=row['format_relaxed_accuracy'];lo,hi=row['format_relaxed_interval']
                    ax.errorbar(index+.12,relaxed,yerr=[[relaxed-lo],[hi-relaxed]],fmt='D',mfc='white',mec='black',ecolor='gray',capsize=3,label='Relaxed numeric match (post-hoc)' if index==0 else None)
                    if relaxed!=p:ax.annotate(f'{row["format_relaxed_correct"]}/{row["received"]}',(index+.12,relaxed),xytext=(6,0),textcoords='offset points',ha='left',fontsize=8)
        ax.set(xticks=range(3),xticklabels=STAGES,ylim=(-.025,1.1),yticks=[0,.2,.4,.6,.8,1],title=category.capitalize(),ylabel='Protocol pass rate (95% Wilson interval)')
        if category=='math':
            ax.set_title(f'Math: strict / relaxed (expected n={row["expected"]})')
            if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=7)
        else:ax.set_title(f'{category.capitalize()} (expected n={row["expected"]})')
        if category=='logic' and report.get('logic_constant_answer_baseline'):
            baseline=report['logic_constant_answer_baseline']
            ax.axhline(baseline['accuracy'],ls='--',color='gray',label=f'Constant "{baseline["answer"]}" baseline')
            ax.legend(fontsize=7)
    fig.suptitle('English quick diagnostic | Real checkpoint outputs | Open-task ratings pending' if report['human_status']=='pending' else 'English quick diagnostic | Real checkpoint outputs')
    fig.savefig(Path(run_dir)/'capabilities.png',dpi=180);fig.savefig(Path(run_dir)/'capabilities.svg');plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for ax,category in zip(axes.flat,['math','logic','code','concept','writing']):
        conditions={'math':['small','large'],'logic':['short','long'],'code':['short','long'],'concept':['closed','context'],'writing':['one','three']}[category]
        labels={'small':'Small numbers','large':'Large numbers','short':'256 tokens' if category=='code' else 'Short chain','long':'512 tokens' if category=='code' else 'Long chain','closed':'No supplied facts','context':'Supplied facts','one':'1 constraint','three':'3 constraints'}
        for si,stage in enumerate(STAGES):
            for ci,condition in enumerate(conditions):
                rows=[c for p in report['boundaries'] if p['category']==category and p['stage']==stage for c in p['conditions'] if c['condition']==condition]
                if category=='concept':values=[r['keyword_coverage'] for r in rows if 'keyword_coverage' in r]
                elif category=='math':values=[float(r['format_relaxed_correct']) for r in rows if 'format_relaxed_correct' in r]
                else:values=[float(r['correct']) for r in rows if r['correct'] is not None]
                if values:
                    x=si+(ci-.5)*.3;mean=sum(values)/len(values)
                    ax.bar(x,mean,width=.3,color=['#2563eb','#0891b2'][ci],label=labels[condition] if si==0 else None)
                    value=f'{mean:.2f}' if category=='concept' else f'{int(sum(values))}/{len(values)}'
                    ax.text(x,mean+.025,value,ha='center',fontsize=7)
        titles={'math':'Math | numeric match (relaxed)','logic':'Logic | exact-format pass','code':'Code | unit-test pass','concept':'Concept | keyword coverage','writing':'Writing | format constraints'}
        ax.set(xticks=range(3),xticklabels=STAGES,ylim=(0,1.1),title=titles[category])
        if ax.get_legend_handles_labels()[0]:ax.legend(fontsize=8,loc='lower left' if category=='concept' else 'upper left' if category=='math' else 'upper right')
    axes.flat[-1].axis('off');axes.flat[-1].text(0,.8,'4 pairs per category\nCounts: passes / measured inputs\nConcept: mean lexical coverage only\nControlled probes, not benchmark scores\nMissing code runs remain unscored',va='top',fontsize=10)
    fig.savefig(Path(run_dir)/'boundaries.png',dpi=180);fig.savefig(Path(run_dir)/'boundaries.svg');plt.close(fig)
    if report['human_status']=='scored':
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,category in zip(axes,['concept','writing']):
            for si,stage in enumerate(STAGES):
                row=next(r for r in report['human_summary'] if r['stage']==stage and r['category']==category)
                raw=[r for r in report['human_records'] if not r.get('is_repeat') and r['stage']==stage and r['task_id'].startswith(category+'-')]
                if raw:
                    color=['#64748b','#0891b2','#2563eb'][si]
                    boxes=ax.boxplot([[r['ratings'][i] for r in raw] for i in range(3)],positions=[i+(si-1)*.23 for i in range(3)],widths=.19,manage_ticks=False,patch_artist=True,showmeans=True)
                    for box in boxes['boxes']:box.set_facecolor(color);box.set_alpha(.5)
                    ax.plot([],[],color=color,label=f'{stage} (n={row["n"]})')
            labels=['Accuracy','Completeness','Clarity'] if category=='concept' else ['Completion','Coherence','Expression']
            ax.set(xticks=range(3),xticklabels=labels,ylim=(.8,5.2),yticks=range(1,6),title=category)
            ax.legend()
        fig.suptitle('Single-reviewer rating distributions | 1-5 scale | Repeats excluded')
        fig.savefig(Path(run_dir)/'human-ratings.png',dpi=180);fig.savefig(Path(run_dir)/'human-ratings.svg');plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','generate','score','review','import-review','analyze','render'])
    parser.add_argument('--suite-dir',type=Path,default=Path('outputs/capability/suite'))
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/capability/final-run'))
    parser.add_argument('--train-dir',type=Path)
    parser.add_argument('--gsm-data',type=Path)
    parser.add_argument('--humaneval-data',type=Path)
    parser.add_argument('--stage',choices=STAGES,default='base');parser.add_argument('--phase',choices=['smoke','formal'],default='formal')
    parser.add_argument('--open-only',action='store_true',help='Generate only main open tasks; later resume without this flag')
    parser.add_argument('--execute-code',action='store_true');parser.add_argument('--ratings',type=Path)
    args=parser.parse_args(argv)
    try:
        if args.command in ('prepare','generate'):
            local=read('outputs/capability/local-inputs.json') if Path('outputs/capability/local-inputs.json').exists() else {}
            if args.train_dir is None:
                if not local.get('train_dir'):raise ValueError('Provide --train-dir or configure outputs/capability/local-inputs.json')
                args.train_dir=Path(local['train_dir'])
            if args.command=='prepare' and args.gsm_data is None:
                if not local.get('gsm_data'):raise ValueError('Provide --gsm-data or configure outputs/capability/local-inputs.json')
                args.gsm_data=Path(local['gsm_data'])
        if args.command=='prepare':prepare(args.gsm_data,args.train_dir,args.suite_dir,args.humaneval_data)
        elif args.command=='generate':generate(args.suite_dir,args.run_dir,args.stage,args.train_dir,args.phase,args.open_only)
        elif args.command=='score':score(args.suite_dir,args.run_dir,args.execute_code,progress=True)
        elif args.command=='review':blind_review(args.suite_dir,args.run_dir)
        elif args.command=='import-review':import_review(args.run_dir,args.ratings)
        elif args.command=='analyze':analyze(args.suite_dir,args.run_dir)
        else:render_report(args.run_dir)
    except (OSError,ValueError,KeyError,subprocess.TimeoutExpired) as exc:parser.error(str(exc))


if __name__=='__main__':main()
