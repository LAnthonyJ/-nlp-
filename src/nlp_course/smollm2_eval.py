"""Run the three frozen SmolLM2 public tasks on six local Qwen checkpoints."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time

import torch

from .capability import read, sha
from .capability_data import digest
from .smollm2_data import assets, freeze, prepare
from .training import TEMPLATE, prompt_text


def spec_for(config, name):
    if name in config["models"]:
        return config["models"][name]
    return dict(config["models"]["base"], adapter=config["adapters"][name], template="project")


def template_prompt(tokenizer, kind, prompt):
    if kind == "raw":
        return prompt
    text=tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False,add_generation_prompt=True)
    if kind=="project" and text!=prompt_text(prompt):
        raise ValueError("Project tokenizer did not render the training template exactly once")
    return text


def load(config, name):
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file

    spec=spec_for(config,name)
    snapshot=Path(snapshot_download(spec["repo"],revision=spec["revision"],local_files_only=True))
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    tokenizer.padding_side="left"
    tokenizer.truncation_side="left"
    tokenizer.pad_token=tokenizer.eos_token
    if spec["template"]=="project":
        tokenizer.chat_template=TEMPLATE
    model=AutoModelForCausalLM.from_pretrained(snapshot,torch_dtype=torch.bfloat16,local_files_only=True).to("cuda")
    weights={p.name:sha(p) for p in sorted(snapshot.glob("*.safetensors"))}
    archived=read("docs/repair-results/protocols.json")["instruct" if name=="instruct" else "base"]["weights"]
    if list(weights.values())!=[archived["base"]]:
        raise ValueError("Base weights differ from the registered Qwen checkpoint")
    if "adapter" in spec:
        folder=Path(spec["adapter"])
        state_path=folder/"adapter_model.safetensors"
        expected=read("docs/paired-dpo-results/metrics.json")["weights"][name]
        if sha(state_path)!=expected:
            raise ValueError(f"Changed selected adapter {name}")
        settings=read(folder/"adapter_config.json")
        # The original adapters are standard attention LoRA saved by a newer PEFT.
        # Reconstruct their active fields and verify every stored tensor after loading.
        fields=("r","lora_alpha","lora_dropout","target_modules","task_type","bias","fan_in_fan_out","use_rslora","use_dora","rank_pattern","alpha_pattern")
        model=get_peft_model(model,LoraConfig(**{k:settings[k] for k in fields},inference_mode=True))
        state=load_file(str(state_path))
        set_peft_model_state_dict(model,state)
        loaded=get_peft_model_state_dict(model)
        if loaded.keys()!=state.keys() or any(not torch.equal(t.cpu(),state[k]) for k,t in loaded.items()):
            raise ValueError(f"Adapter tensor verification failed for {name}")
        weights.update(adapter=expected,adapter_config=sha(folder/"adapter_config.json"))
    protocol=dict(spec=spec, weights=weights, tokenizer=digest(tokenizer.get_vocab()),
                  tokenizer_config=sha(snapshot/"tokenizer_config.json"),model_config=sha(snapshot/"config.json"),
                  generation=sha(snapshot/"generation_config.json"),template=tokenizer.chat_template if spec["template"]!="raw" else None,
                  eos=tokenizer.eos_token_id,pad=tokenizer.pad_token_id)
    return model.eval(),tokenizer,protocol


class PairEncoder:
    """Use the pinned backend's joint tokenization, including boundary whitespace."""
    def __init__(self, tokenizer):
        self.tokenizer=tokenizer
        self.add_special_tokens=True

    def tok_encode(self,text,add_special_tokens=None):
        return self.tokenizer.encode(text,add_special_tokens=True if add_special_tokens is None else add_special_tokens)

    def pair(self,context,choice):
        from lighteval.models.abstract_model import LightevalModel
        return LightevalModel.tok_encode_pair(self,context,choice,pairwise=False)


def choice_results(model,tok,rows,kind):
    from lighteval.metrics.normalizations import normalize_log_probs, LogProbCharNorm
    from lighteval.metrics.metrics import Metrics
    from lighteval.tasks.requests import Doc
    encoder=PairEncoder(tok)
    encoded=[]
    for r in rows:
        text=template_prompt(tok,kind,r["prompt"])
        encoded.extend(encoder.pair(text,choice) for choice in r["choices"])
    scores=[]
    # Only score the continuation tokens. Cap padded tokens per forward on 6 GB.
    i=0
    while i<len(encoded):
        end=i+1
        while end<len(encoded) and (end+1-i)*max(len(c)+len(a) for c,a in encoded[i:end+1])<=1024:
            end+=1
        chunk=encoded[i:end]
        sequences=[c+a for c,a in chunk]
        width=max(map(len,sequences))-1
        if width>=2048 or any(not c or not a for c,a in chunk):
            raise ValueError("Unsupported empty or overlong candidate")
        ids=torch.tensor([s[:-1]+[tok.pad_token_id]*(width+1-len(s)) for s in sequences],device="cuda")
        logits=model(input_ids=ids,use_cache=False).logits
        for j,(c,a) in enumerate(chunk):
            segment=logits[j,len(c)-1:len(c)+len(a)-1]
            values=segment.log_softmax(-1).gather(1,torch.tensor(a,device="cuda")[:,None])
            scores.append(values.sum().item())
        del logits
        i=end
    output=[]
    start=0
    for row in rows:
        choices=row["choices"]
        raw=scores[start:start+len(choices)]
        start+=len(choices)
        normalized=normalize_log_probs(LogProbCharNorm(ignore_first_space=True),raw,None,choices,None)
        pred=max(range(len(choices)),key=normalized.__getitem__)
        doc=Doc(query=row["prompt"],choices=choices,gold_index=row["gold"])
        correct=Metrics.loglikelihood_acc_norm_nospace.value.sample_level_fn(
            gold_ixs=[row["gold"]],choices_logprob=raw,unconditioned_logprob=None,choices_tokens=None,formatted_doc=doc)
        if bool(correct)!=(pred==row["gold"]) or not all(map(math.isfinite,raw)):
            raise ValueError("Invalid candidate likelihood or inconsistent prediction")
        output.append(dict(id=row["id"],gold=row["gold"],pred=pred,logprob=raw,normalized=normalized,correct=bool(correct)))
    return output


def score_instruction(row,answer):
    from lighteval.tasks.extended.ifeval.main import ifeval_metric,ifeval_prompt
    doc=ifeval_prompt(dict(prompt=row["prompt"],instruction_id_list=row["instructions"],kwargs=row["kwargs"]))
    return ifeval_metric(predictions=[answer],formatted_doc=doc)


def instruction_results(model,tok,rows,kind,budget):
    texts=[template_prompt(tok,kind,row["prompt"]) for row in rows]
    batch=tok(texts,padding=True,truncation=True,max_length=2047,return_tensors="pt").to("cuda")
    budget=min(budget,2048-batch.input_ids.size(1))
    tokens=model.generate(**batch,do_sample=False,max_new_tokens=budget,eos_token_id=tok.eos_token_id,
                          pad_token_id=tok.pad_token_id,use_cache=True)[:,batch.input_ids.size(1):].tolist()
    result=[]
    for row,ids in zip(rows,tokens,strict=True):
        stop=ids.index(tok.eos_token_id) if tok.eos_token_id in ids else len(ids)
        answer=tok.decode(ids[:stop],skip_special_tokens=True)
        scores=score_instruction(row,answer)
        result.append(dict(id=row["id"],response=answer,output_tokens=min(stop+1,len(ids)),truncated=stop==len(ids),
                           scores=scores,correct=bool(scores["prompt_level_strict_acc"])))
    return result


def connect(path):
    db=sqlite3.connect(path)
    db.execute("CREATE TABLE IF NOT EXISTS results (protocol TEXT, id TEXT, record TEXT NOT NULL, PRIMARY KEY(protocol,id))")
    db.execute("CREATE TABLE IF NOT EXISTS costs (protocol TEXT, n INTEGER, seconds REAL, peak INTEGER)")
    return db


def ordered(db,key,rows,complete=True):
    indexed={i:json.loads(data) for i,data in db.execute("SELECT id,record FROM results WHERE protocol=?",(key,))}
    ids=[row["id"] for row in rows]
    if not indexed.keys()<=set(ids) or complete and set(ids)!=indexed.keys():
        raise ValueError("Result IDs differ from the frozen split")
    if any(row["id"]!=i for i,row in indexed.items()):
        raise ValueError("Corrupt stored result ID")
    return [indexed[i] for i in ids if i in indexed]


def commit_batch(db,key,rows,seconds,peak):
    with db:
        db.executemany("INSERT INTO results VALUES (?,?,?)",[(key,r["id"],json.dumps(r,ensure_ascii=False,allow_nan=False)) for r in rows])
        db.execute("INSERT INTO costs VALUES (?,?,?,?)",(key,len(rows),seconds,peak))


@torch.inference_mode()
def evaluate(config,name,task,mode,batch_size):
    root=Path(config["root"])
    rows=read(root/f"{task}.json")
    if mode=="check":
        rows=rows[:8]
    if name=="base" and task=="ifeval":
        raise ValueError("IFEval is outside the SmolLM2 base protocol")
    model,tok,info=load(config,name)
    kind=info["spec"]["template"]
    versions={pkg:importlib.metadata.version(pkg) for pkg in ("torch","transformers","tokenizers","peft","lighteval","langdetect","nltk")}
    protocol=dict(model=info,task=task,mode=mode,data=digest(rows),settings=config["tasks"][task],versions=versions,
                  upstream=config["upstream"],lighteval=config["lighteval"],dtype="bfloat16",backend="transformers",
                  batch=batch_size,seed=config["seed"],device=torch.cuda.get_device_name(),n=len(rows),
                  code=sha(Path(__file__)),sample_prompt=template_prompt(tok,kind,rows[0]["prompt"]))
    if mode=="check" and task=="ifeval":
        known=dict(prompt="Use lowercase.",instructions=["change_case:english_lowercase"],kwargs=[{}])
        if not score_instruction(known,"this is a complete sentence in english.")["prompt_level_strict_acc"] or score_instruction(known,"THIS IS UPPERCASE.")["prompt_level_strict_acc"]:
            raise ValueError("IFEval controls failed")
    key=digest(protocol)
    freeze(root/mode/task/f"{name}.json",dict(key=key,**protocol))
    db=connect(root/f"{mode}.sqlite")
    try:
        done={r["id"] for r in ordered(db,key,rows,complete=False)}
        pending=[r for r in rows if r["id"] not in done]
        # Similar prompt lengths reduce left padding for greedy generation.
        for pos in range(0,len(pending),batch_size):
            batch=pending[pos:pos+batch_size]
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            before=time.perf_counter()
            result=(instruction_results(model,tok,batch,kind,config["tasks"][task]["budget"]) if task=="ifeval"
                    else choice_results(model,tok,batch,kind))
            torch.cuda.synchronize()
            commit_batch(db,key,result,time.perf_counter()-before,torch.cuda.max_memory_allocated())
            if pos//batch_size%50==0:
                print(mode,task,name,len(done)+pos+len(batch),"/",len(rows),flush=True)
        all_rows=ordered(db,key,rows)
        print(mode,task,name,sum(r["correct"] for r in all_rows),"/",len(rows),flush=True)
    finally:
        db.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","check","run","report"))
    parser.add_argument("--config",type=Path,default=Path("configs/smollm2-public.json"))
    parser.add_argument("--task",choices=("piqa","hellaswag","ifeval"))
    parser.add_argument("--model")
    parser.add_argument("--batch",type=int)
    parser.add_argument("--output",type=Path,default=Path("docs/smollm2-results"))
    args=parser.parse_args()
    cfg=read(args.config)
    cache=Path(cfg["assets"]).resolve()
    os.environ.setdefault("NLTK_DATA",str(cache/"nltk_data"))
    # The historical registry eagerly loads an ancillary BLEURT model at import.
    if (cache/"hub").is_dir():
        os.environ.setdefault("TRANSFORMERS_CACHE",str(cache/"hub"))
    if args.command=="prepare":
        return prepare(cfg)
    if args.command=="report":
        from .smollm2_report import export
        return export(cfg,args.output)
    from langdetect import DetectorFactory
    DetectorFactory.seed=cfg["seed"]
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(1)
    batch=args.batch or cfg["batch"]
    if batch<1 or not torch.cuda.is_available():
        raise ValueError("Positive batch size and CUDA required")
    for task in [args.task] if args.task else cfg["tasks"]:
        for name in [args.model] if args.model else [*cfg["models"],*cfg["adapters"]]:
            if task=="ifeval" and name=="base":
                continue
            evaluate(cfg,name,task,args.command,batch)
            gc.collect()
            torch.cuda.empty_cache()


if __name__=="__main__":
    main()
