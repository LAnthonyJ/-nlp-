"""Versioned repair experiments using the existing isolated capability scorer."""
from collections import Counter
import gzip
import json
from pathlib import Path
import random

from . import capability as cap
from .capability_data import digest
from .training import synthetic_rows


def freeze(config):
    destination = Path(config['suite_dir'])
    if destination.exists():
        raise ValueError('Suite directory already exists')
    sources = config['sources']
    old = cap.read(sources['old_suite'])['tasks']
    excluded = {t['id'] for t in old}
    rng = random.Random(42)
    gsm = [json.loads(line) for line in Path(sources['gsm_test']).read_text(encoding='utf-8').splitlines() if line.strip()]
    code = [json.loads(line) for line in gzip.decompress(Path(sources['humaneval']).read_bytes()).decode().splitlines() if line.strip()]
    if len(gsm)!=1319 or len(code)!=164:
        raise ValueError('Expected complete official GSM8K test and HumanEval sources')
    tasks = []
    for i in rng.sample([i for i in range(len(gsm)) if f'gsm-{i}' not in excluded],30):
        row=gsm[i]
        tasks.append(dict(id=f'gsm-{i}',category='math',prompt=row.get('question',row.get('query')),reference=row.get('answer',row.get('response')).split('####')[-1].strip(),split='main'))
    for row in rng.sample([r for r in code if r['task_id'].replace('/','-') not in excluded],25):
        tasks.append(dict(id=row['task_id'].replace('/','-'),category='code',prompt='Complete this Python function. Return the complete function in one Python code block.\n'+row['prompt'],reference=row['canonical_solution'],tests=row['test'],entry_point=row['entry_point'],split='main'))
    for i in range(10):
        yes=i%2==0
        tasks.append(dict(id=f'logic-judgment-{i}',category='logic',kind='entailment',prompt=f'All copper objects are heavy. Object {i} is {"copper" if yes else "wooden"}. Does it necessarily follow that object {i} is heavy? Answer yes or no only.',reference='yes' if yes else 'no',split='main'))
        names=[f'N{j+i*5}' for j in range(4)];rng.shuffle(names)
        facts=[f'{a} comes before {b}.' for a,b in zip(names,names[1:])];rng.shuffle(facts)
        tasks.append(dict(id=f'logic-order-{i}',category='logic',kind='ordering',prompt=' '.join(facts)+' Return the complete order using comma-separated names only.',reference=','.join(names),split='main'))
        tasks.append(dict(id=f'logic-count-{i}',category='logic',kind='counting',prompt=f'A box contains {17+i} green marbles and {9+i} white marbles. Remove {i+2} green marbles. How many marbles remain? Return only an integer.',reference=str(24+i),split='main'))
    concepts=['evaporation','photosynthesis','an algorithm','a database index','a compiler','a neural network activation','a probability distribution','a renewable resource','a feedback loop','a control group']
    for i,topic in enumerate(concepts):
        tasks.append(dict(id=f'concept-new-{i}',category='concept',prompt=f'Explain {topic} for a beginning student in two sentences.',reference='Rate factual accuracy, completeness, and clarity from 1 to 5 using the existing anchored rubric.',split='main'))
    for i,topic in enumerate(['a train trip','cooking soup','learning chess','saving water','a garden','a museum','a mountain walk','a classroom','a bicycle','a rainy morning']):
        tasks.append(dict(id=f'writing-new-{i}',category='writing',prompt=f'Write a coherent three-sentence paragraph about {topic}. The last sentence must start with "Finally,".',reference='Rate task completion, coherence, and expression from 1 to 5 using the existing anchored rubric.',split='main'))
    dev=[]
    train_gsm=[json.loads(line) for line in Path(sources['gsm_train']).read_text(encoding='utf-8').splitlines() if line.strip()]
    for i in rng.sample(range(len(train_gsm)),10):
        r=train_gsm[i]
        dev.append(dict(id=f'dev-gsm-{i}',category='math',prompt=r['question'],reference=r['answer'].split('####')[-1].strip(),split='main'))
    for i in range(10):
        dev.append(dict(id=f'dev-logic-{i}',category='logic',prompt=f'All silver objects are cold. Object {i} is {"silver" if i%2==0 else "red"}. Must it be cold? Answer yes or no only.',reference='yes' if i%2==0 else 'no',split='main'))
        name=f'offset_{i}'
        dev.append(dict(id=f'dev-code-{i}',category='code',prompt=f'Return a complete Python function {name}(x) that returns x plus {i+3}, in one Python code block.',entry_point=name,tests=f'def check(f):\n    assert f(0)=={i+3}\n    assert f(-7)=={-4+i}',reference=f'def {name}(x): return x + {i+3}',split='main'))
    for folder,rows in [(destination,tasks),(Path(config['dev_suite_dir']),dev)]:
        if (folder/'suite.json').exists(): raise ValueError('Refusing to replace a frozen suite')
        cap.save(folder/'suite.json',dict(schema_version=2,seed=42,tasks=rows,near_overlap_candidates=[],sources={k:cap.sha(v) for k,v in sources.items()},task_hash=digest(rows)))


def repair_tail(path):
    """Discard only an interrupted final append; interior corruption is an error."""
    if not path.exists(): return
    lines=path.read_bytes().splitlines(keepends=True)
    for i,line in enumerate(lines):
        try: json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if i!=len(lines)-1 or line.endswith(b'\n'): raise ValueError('Corrupt completed record')
            path.write_bytes(b''.join(lines[:i]));return
    if lines and not lines[-1].endswith(b'\n'):
        with path.open('ab') as stream: stream.write(b'\n')


def run(config, command, model=None, execute_code=False):
    if command not in {'freeze','generate','score','review','analyze','render'}:
        raise ValueError('Unsupported experiment command: '+command)
    if config.get('statistics_only') and command not in {'analyze','render'}:
        raise ValueError('This public configuration contains statistics only; use the local experiment for generation or scoring')
    suite=Path(config['suite_dir']);out=Path(config['run_dir'])
    models=config['models']
    if len(models)!=len(set(models)) or any(not key.replace('_','').isalnum() for key in models):
        raise ValueError('Model identifiers must be safe filenames')
    if command=='freeze': return freeze(config)
    if command=='generate':
        if model and model not in models:raise ValueError('Unknown model name')
        for name,spec in models.items():
            if model and name!=model: continue
            repair_tail(out/'formal'/f'{name}.jsonl')
            cap.generate(suite,out,name,None,phase='smoke',spec=spec)
            cap.generate(suite,out,name,None,spec=spec)
        return
    tasks=cap.read(suite/'suite.json')['tasks']
    ids={t['id'] for t in tasks}
    scored=[]
    if command=='score':
        for name in models:
            rows=cap.records(out/'formal'/f'{name}.jsonl')
            if len(rows)!=len(ids) or {r['task_id'] for r in rows}!=ids or any(r['stage']!=name for r in rows):
                raise ValueError(f'Incomplete or inconsistent model records: {name}')
            by_id={t['id']:t for t in tasks}
            scored.extend(cap.score_task(by_id[r['task_id']],r,execute_code) for r in rows)
        cap.save(out/'scores.json',dict(records=scored,experiment_hash=digest(config)))
        return
    if command=='review':
        original=cap.STAGES
        try:
            cap.STAGES=tuple(models)
            return cap.blind_review(suite,out)
        finally: cap.STAGES=original
    scores=cap.read(out/'scores.json')
    if scores['experiment_hash']!=digest(config): raise ValueError('Experiment changed after scoring')
    summary=[]
    for name in models:
        for category in ['math','logic','code','concept','writing']:
            rows=[r for r in scores['records'] if r['stage']==name and r['category']==category]
            measured=[r for r in rows if r['correct'] is not None]
            good=sum(r['correct'] for r in measured)
            summary.append(dict(model=name,category=category,n=len(rows),measured=len(measured),correct=good,accuracy=good/len(measured) if measured else None,interval=cap.wilson(good,len(measured)),relaxed_correct=sum(r.get('format_relaxed_correct',False) for r in rows),truncated=sum(r.get('truncated',False) for r in rows),statuses=dict(Counter(r['status'] for r in rows))))
    transitions=[]
    lookup={(r['stage'],r['task_id']):r for r in scores['records']}
    for first,last in config.get('comparisons',[]):
        for task in tasks:
            a,b=lookup[first,task['id']],lookup[last,task['id']]
            if a['correct'] is not None and b['correct'] is not None:
                transitions.append(dict(first=first,last=last,task_id=task['id'],before=a['correct'],after=b['correct']))
    human=cap.read(out/'human-scores.json') if (out/'human-scores.json').exists() else []
    by_id={t['id']:t for t in tasks}
    human_summary=[]
    for name in models:
        for category in ['concept','writing']:
            rows=[r for r in human if not r['is_repeat'] and r['stage']==name and by_id[r['task_id']]['category']==category]
            human_summary.append(dict(model=name,category=category,n=len(rows),means=[sum(r['ratings'][i] for r in rows)/len(rows) for i in range(3)] if rows else None))
    baseline=Counter(t['reference'] for t in tasks if t['category']=='logic')
    logic_by_kind=[]
    for name in models:
        for kind in sorted({t.get('kind','unspecified') for t in tasks if t['category']=='logic'}):
            selected={t['id']:t for t in tasks if t['category']=='logic' and t.get('kind','unspecified')==kind}
            rows=[r for r in scores['records'] if r['stage']==name and r['task_id'] in selected]
            references=Counter(t['reference'] for t in selected.values())
            logic_by_kind.append(dict(model=name,kind=kind,n=len(rows),correct=sum(r['correct'] is True for r in rows),constant_answer_accuracy=max(references.values())/len(selected)))
    cap.save(out/'analysis.json',dict(summary=summary,transitions=transitions,experiment=config,human_summary=human_summary,human_status='scored' if human else 'pending',logic_by_kind=logic_by_kind,logic_constant_baseline=dict(counts=baseline,accuracy=max(baseline.values())/sum(baseline.values()) if baseline else None)))
    if command=='render':
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for ax,category in zip(axes,['math','logic','code']):
            rows=[r for r in summary if r['category']==category]
            for i,r in enumerate(rows):
                if r['accuracy'] is None: ax.text(i,.1,'Unscored',rotation=90)
                else:
                    p=r['accuracy'];lo,hi=r['interval']
                    ax.bar(i,p,color='#0891b2');ax.errorbar(i,p,yerr=[[p-lo],[hi-p]],color='black',capsize=3)
                    ax.text(i,hi+.04,f'{r["correct"]}/{r["measured"]}',ha='center')
                if category=='math' and r['n']:
                    numeric=r['relaxed_correct']/r['n']
                    low,high=cap.wilson(r['relaxed_correct'],r['n'])
                    ax.errorbar(i+.12,numeric,yerr=[[numeric-low],[high-numeric]],fmt='D',mfc='white',mec='black',ecolor='gray',capsize=3,label='Explicit final numeric answer' if i==0 else None)
                    if numeric!=r['accuracy']:
                        ax.text(i+.12,high+.04,f'{r["relaxed_correct"]}/{r["n"]}',ha='center')
            titles={'math':'Math: strict bars / numeric diamonds','logic':'Logic: strict format','code':'Code: isolated tests'}
            ax.set(title=titles[category],ylim=(0,1.1),xticks=range(len(models)),xticklabels=list(models));ax.tick_params(axis='x',rotation=35)
            if category=='math':ax.legend(fontsize=8,loc='upper left')
        axes[0].set_ylabel('Fraction correct (95% Wilson interval)')
        fig.savefig(out/'capabilities.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
        statuses=sorted({s for r in summary for s in r['statuses']})
        bottom=[0]*len(models)
        colors={'correct':'#0f766e','incorrect':'#f59e0b','incorrect_or_format':'#fb923c','invalid_answer':'#facc15','pending_human':'#cbd5e1','syntax_error':'#be123c','test_failure':'#f87171'}
        labels={'correct':'Passed','incorrect':'Wrong numeric answer','incorrect_or_format':'Logic answer / format','invalid_answer':'No valid final answer','pending_human':'Human rating pending','syntax_error':'Code syntax error','test_failure':'Code test failure'}
        for status in statuses:
            values=[sum(r['statuses'].get(status,0) for r in summary if r['model']==name) for name in models]
            ax.bar(list(models),values,bottom=bottom,label=labels.get(status,status),color=colors.get(status,'#94a3b8'));bottom=[a+b for a,b in zip(bottom,values)]
        ax.legend(fontsize=8,loc='upper left',bbox_to_anchor=(1.01,1));ax.set_ylabel('Responses');ax.set_title('Response scoring status');fig.savefig(out/'errors.png',dpi=180);plt.close(fig)
