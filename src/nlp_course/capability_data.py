"""Fixed, independently scored capability tasks; never used for training."""
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import urllib.request

CONCEPTS = [
 ('What is monetary policy?', ['central bank', 'interest', 'inflation'], 'Monetary policy is conducted by a central bank; interest rates affect borrowing and demand; price stability is a common goal.'),
 ('Explain the difference between inflation and a high price level.', ['increase', 'price', 'time'], 'Inflation is the rate of increase in the price level over time; a high price level alone does not imply ongoing inflation.'),
 ('Explain opportunity cost using a simple example.', ['alternative', 'choice'], 'Opportunity cost is the value of the best alternative forgone when a choice is made.'),
 ('Explain what diversification can and cannot do to investment risk.', ['risk', 'market'], 'Diversification can reduce asset-specific risk; common market risk remains.'),
 ('Explain the uncertainty principle without describing it as a faulty measuring device.', ['position', 'momentum'], 'Position and momentum have a fundamental uncertainty tradeoff; it is not merely faulty measuring equipment.'),
 ('Explain why seasons occur on Earth.', ['tilt', 'sunlight'], 'Earth has an axial tilt; seasonal changes in sunlight angle and day length cause seasons, rather than Earth-Sun distance.'),
 ('Explain the difference between correlation and causation.', ['cause', 'variable'], 'Association alone does not establish cause; a third variable or reverse causation may explain it.'),
 ('Explain utilitarianism and one limitation.', ['consequence', 'welfare'], 'Utilitarianism evaluates consequences for aggregate welfare; individual rights can conflict with aggregation.'),
 ('Explain one way Enlightenment ideas influenced the American Revolution.', ['rights', 'government'], 'Ideas of natural rights and government by consent influenced revolutionary political arguments.'),
 ('Explain the difference between supervised learning and reinforcement learning.', ['label', 'reward'], 'Supervised learning uses labeled examples; reinforcement learning learns behavior through interaction and rewards.'),
]
WRITING = [
 'Summarize in two sentences: A town replaced old streetlights with efficient LEDs. Energy use fell, but residents reported excessive brightness. The council installed shields and adjusted nighttime intensity.',
 'Summarize in two sentences: A school tested free breakfasts for one semester. Attendance improved, but the study lacked random assignment. Researchers recommended a larger controlled trial.',
 'Summarize in two sentences: A library began lending tools. Demand exceeded forecasts. Staff introduced reservations and maintenance checks, while retaining free access.',
 'Rewrite politely without changing the request: Your report is late. Send it today and explain the delay.',
 'Rewrite for a general audience: Parameter-efficient adaptation optimizes a low-dimensional update while freezing pretrained parameters.',
 'Explain how to plan a study session using exactly three numbered steps.',
 'Write a short explanation of composting with sections titled Benefits and Limitations.',
 'Write a short balanced comparison of remote and classroom learning, mentioning one benefit and one drawback of each.',
 'Write a short story about a lost key that ends with the sentence: The door was already open.',
 'Write a short story about cooperation between two neighbors, without introducing a third character.',
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def prepare(gsm_path, train_dir, destination, humaneval_path=None):
    rng = random.Random(42)
    gsm = [json.loads(s) for s in Path(gsm_path).read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(gsm) != 1319:
        raise ValueError('Expected the 1,319-example GSM8K test split')
    if humaneval_path:
        raw = Path(humaneval_path).read_bytes()
    else:
        url = 'https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz'
        raw = urllib.request.urlopen(url, timeout=60).read()
    if raw[:2] == b'\x1f\x8b':
        raw = gzip.decompress(raw)
    code = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    tasks = []
    for index in rng.sample(range(len(gsm)), 30):
        row = gsm[index]
        tasks.append(dict(id=f'gsm-{index}', category='math', prompt=row['query'], reference=row['response'].split('####')[-1].strip(), source='GSM8K test subset', split='main'))
    for row in rng.sample(code, 25):
        tasks.append(dict(id=row['task_id'].replace('/', '-'), category='code', prompt='Complete this Python function. Return the complete function in one Python code block.\n'+row['prompt'], reference=row['canonical_solution'], tests=row['test'], entry_point=row['entry_point'], source='HumanEval subset', split='main'))
    for i in range(25):
        names = ['A', 'B', 'C', 'D', 'E'][:3+i%3]
        rng.shuffle(names)
        facts = ' '.join(f'{a} comes before {b}.' for a,b in zip(names,names[1:]))
        prompt=f'{facts} Return the complete order using comma-separated letters only.'
        reference=','.join(names)
        kind=i%5
        if kind==1:
            prompt=f'All amber objects are blue. All blue objects are cold. Object {i} is amber. Must it be cold? Answer yes or no only.';reference='yes'
        elif kind==2:
            prompt=f'Set A is {{{i}, {i+1}, {i+2}}}. Set B is {{{i+1}, {i+2}, {i+3}}}. How many elements are in their intersection? Return only an integer.';reference='2'
        elif kind==3:
            prompt=f'There are {i+2} red balls and {i+3} blue balls. Two red balls are removed. How many balls remain? Return only an integer.';reference=str(2*i+3)
        elif kind==4:
            prompt=f'If the lamp is on, the switch is closed. The lamp is on. Is it logically necessary that the switch is closed? Answer yes or no only.';reference='yes'
        tasks.append(dict(id=f'logic-{i}',category='logic',prompt=prompt,reference=reference,source='Generated logical oracle',split='main'))
    for i,(prompt,keywords,facts) in enumerate(CONCEPTS):
        tasks.append(dict(id=f'concept-{i}',category='concept',prompt=prompt,reference=facts,keywords=keywords,source='Project-authored concept rubric',split='main'))
    for i,prompt in enumerate(WRITING):
        tasks.append(dict(id=f'writing-{i}',category='writing',prompt=prompt,reference='Score task completion, coherence, expression on the anchored rubric.',source='Project-authored writing rubric',split='main'))
    # Twenty paired interventions, kept separate from the 100 main tasks.
    for i in range(4):
        a,b = 7+i, 3+i
        for level,scale in [('small',1),('large',1000)]:
            tasks.append(dict(id=f'dmath-{i}-{level}',category='math',pair=f'dmath-{i}',condition=level,split='diagnostic',prompt=f'Compute {a*scale} * {b}. End with The answer is: NUMBER.',reference=str(a*b*scale),source='Programmatic multiplication oracle'))
        for level,n in [('short',3),('long',7)]:
            names=[chr(65+j) for j in range(n)]
            rng.shuffle(names)
            tasks.append(dict(id=f'dlogic-{i}-{level}',category='logic',pair=f'dlogic-{i}',condition=level,split='diagnostic',prompt=' '.join(f'{a} comes before {b}.' for a,b in zip(names,names[1:]))+' Return the complete order using comma-separated letters only.',reference=','.join(names),source='Programmatic ordering oracle'))
        prompt,keywords,facts=CONCEPTS[i]
        for level,context in [('closed',''),('context','Reference facts: '+facts+'\n')]:
            tasks.append(dict(id=f'dconcept-{i}-{level}',category='concept',pair=f'dconcept-{i}',condition=level,split='diagnostic',prompt=context+prompt,reference=facts,keywords=keywords,source='Reference-context intervention'))
        for level in ['one','three']:
            prompt='Explain recycling in exactly three nonempty lines.'
            if level=='three':prompt+=' Start each line with "- ", and end the last line with "Reuse matters."'
            tasks.append(dict(id=f'dwriting-{i}-{level}',category='writing',pair=f'dwriting-{i}',condition=level,split='diagnostic',prompt=prompt+f' Address a household with {i+1} residents.',reference=level,source='Verifiable format constraints'))
        original=[t for t in tasks if t['category']=='code' and t['split']=='main'][i]
        for level,budget in [('short',256),('long',512)]:
            tasks.append(dict(original,id=f'dcode-{i}-{level}',pair=f'dcode-{i}',condition=level,split='diagnostic',max_new_tokens=budget))
    # Conservative prompt leakage screening; flag near overlaps for review.
    norm=lambda s: re.sub(r'\s+',' ',s.casefold()).strip()
    train=[]
    audit={}
    for name in ['train_sft.json','train_dpo.json']:
        rows=json.loads((Path(train_dir)/name).read_text(encoding='utf-8'))
        train.extend(norm(r['instruction']) for r in rows)
        audit[name]=[dict(index=j,record=rows[j]) for j in rng.sample(range(len(rows)),20)]
    train_set=set(train)
    for task in tasks:
        if norm(task['prompt']) in train_set:
            raise ValueError('Exact training/test overlap: '+task['id'])
    near=[]
    tokenized=[set(s.split()) for s in train]
    for task in tasks:
        words=set(norm(task['prompt']).split())
        for index,tokens in enumerate(tokenized):
            if words and len(words&tokens)/len(words|tokens)>=.8:
                near.append(dict(task_id=task['id'],training_index=index))
    report=dict(schema_version=1,seed=42,tasks=tasks,training_audit=audit,near_overlap_candidates=near,
                sources=dict(gsm8k_sha256=digest(gsm),humaneval_sha256=hashlib.sha256(raw).hexdigest()),
                note='Quick English diagnostic; benchmark subsets and project-authored tasks are reported separately.')
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    (destination/'suite.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report
