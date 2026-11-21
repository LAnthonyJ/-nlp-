"""Freeze complete public splits with the task formatters used by SmolLM2."""

import importlib.util
import json
import os
import sys
import tempfile
import types
import urllib.request
import zipfile
import zlib
from pathlib import Path

from .capability import read, save, sha
from .capability_data import digest


# File hashes identify the exact validation data and executable upstream formatters.
def assets(config):
    tasks = config["tasks"]
    upstream = (
        "https://raw.githubusercontent.com/huggingface/smollm/"
        + config["upstream"]
        + "/text/evaluation/smollm2/"
    )
    hub = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/") + "/datasets/"
    return {
        "tasks.py": (
            upstream + "tasks.py",
            "6899e7565b529ef2200e500fd9814c46bbdf42f9576f069d6de8a4dc2b9355b1",
        ),
        "math_utils.py": (
            upstream + "math_utils.py",
            "a8cea086bab6352284a6b88433549f76b7b10047b080f11e2faa6a4a755f0690",
        ),
        "piqa.zip": (
            "https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip",
            "54d32a04f59a7e354396f321723c8d7ec35cc6b08506563d8d1ffcc15ce98ddd",
        ),
        "hellaswag.parquet": (
            hub
            + "Rowan/hellaswag/resolve/"
            + tasks["hellaswag"]["parquet_revision"]
            + "/data/validation-00000-of-00001.parquet",
            "899813071e1e95efafec90f856e1987d2150fa4d020fc005df6962c259f660cd",
        ),
        "ifeval.jsonl": (
            hub
            + "google/IFEval/resolve/"
            + tasks["ifeval"]["revision"]
            + "/ifeval_input_data.jsonl",
            "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
        ),
    }


def ensure_asset(path, url, expected):
    if path.exists() and sha(path) == expected:
        return
    import shutil

    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fp:
        temp = Path(fp.name)
    try:
        with (
            urllib.request.urlopen(url, timeout=120) as response,
            temp.open("wb") as fp,
        ):
            shutil.copyfileobj(response, fp)
        if sha(temp) != expected:
            raise ValueError(f"Upstream file checksum mismatch: {path.name}")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def prepare_registry(folder):
    """Provision the pinned registry's eager BLEURT import and IFEval's tokenizers."""
    revision = "82430b75718a0647fc49b9216ccdd9f0b30dfa72"
    repo = folder / "hub/models--Elron--bleurt-tiny-512"
    snapshot = repo / "snapshots" / revision
    snapshot.mkdir(parents=True, exist_ok=True)
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    for name, checksum in {
        "config.json": "4e3f59ae76eece27a595bd0a0a89176b2d930d28e4a35beb071c7d013713c010",
        "pytorch_model.bin": "bc56c3b8375d540f1ba09636368ad5e53f2237b0671bcf3ff99347739283cac7",
        "special_tokens_map.json": "303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3",
        "tokenizer_config.json": "fedf7ce032f427130ebbae9c7a3a1f46a2dcf11405bf0f243c3926b4c1fe4a72",
        "tokenizer.json": "ad3a0d445d3e88c2f679eb9d309cf0f6dd4e6325f559bb815490c935c57e0d24",
        "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    }.items():
        ensure_asset(
            snapshot / name,
            f"{endpoint}/Elron/bleurt-tiny-512/resolve/{revision}/{name}",
            checksum,
        )
    (repo / "refs").mkdir(exist_ok=True)
    (repo / "refs/main").write_text(revision, encoding="ascii")
    os.environ.setdefault("TRANSFORMERS_CACHE", str((folder / "hub").resolve()))
    for name, checksum in [
        ("punkt", "51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec"),
        (
            "punkt_tab",
            "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
        ),
    ]:
        archive = folder / (name + ".zip")
        ensure_asset(
            archive,
            "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/"
            + archive.name,
            checksum,
        )
        with zipfile.ZipFile(archive) as source:
            target = folder / "nltk_data/tokenizers"
            for member in source.infolist():
                path = target / member.filename
                if member.is_dir() or (
                    path.is_file() and zlib.crc32(path.read_bytes()) == member.CRC
                ):
                    continue
                source.extract(member, target)


def upstream_tasks(folder):
    # SmolLM2's tasks.py uses a relative import of its sibling math_utils.py.
    package = types.ModuleType("_public_smollm2")
    package.__path__ = [str(folder.resolve())]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location(
        "_public_smollm2.tasks", folder / "tasks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {t.name: t for t in module.TASKS_TABLE}


def raw_rows(folder, task):
    if task == "piqa":
        with zipfile.ZipFile(folder / "piqa.zip") as archive:
            rows = archive.read("physicaliqa-train-dev/dev.jsonl").splitlines()
            labels = archive.read("physicaliqa-train-dev/dev-labels.lst").splitlines()
        return [
            dict(json.loads(r), label=int(label))
            for r, label in zip(rows, labels, strict=True)
        ]
    if task == "hellaswag":
        import pyarrow.parquet as pq

        return pq.read_table(folder / "hellaswag.parquet").to_pylist()
    return [
        json.loads(line) for line in (folder / "ifeval.jsonl").read_bytes().splitlines()
    ]


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(
                f"Frozen input changed at {path}; use a fresh output directory"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as fp:
            temp = Path(fp.name)
        try:
            save(temp, value)
            try:
                if os.name == "nt":
                    os.rename(temp, path)  # Atomic, refuses to replace on Windows.
                else:
                    os.link(temp, path)
            except FileExistsError:
                if read(path) != value:
                    raise ValueError(f"Conflicting frozen input at {path}")
        finally:
            temp.unlink(missing_ok=True)


def prepare(config):
    folder = Path(config["assets"])
    folder.mkdir(parents=True, exist_ok=True)
    prepare_registry(folder)
    for name, (url, expected) in assets(config).items():
        ensure_asset(folder / name, url, expected)
    upstream = upstream_tasks(folder)
    root = Path(config["root"])
    for task, settings in config["tasks"].items():
        rows = []
        for index, raw in enumerate(raw_rows(folder, task)):
            if task == "ifeval":
                row = dict(
                    prompt=raw["prompt"],
                    instructions=raw["instruction_id_list"],
                    kwargs=raw["kwargs"],
                )
                source_id = str(raw["key"])
            else:
                definition = upstream[task]
                if definition.hf_revision != settings["revision"]:
                    raise ValueError("Task and dataset revisions do not match")
                doc = definition.prompt_function(raw, task)
                if not 0 <= doc.gold_index < len(doc.choices):
                    raise ValueError("Missing or invalid validation label")
                row = dict(prompt=doc.query, choices=doc.choices, gold=doc.gold_index)
                source_id = str(index)
            rows.append(dict(id=f"{task}-{source_id}", **row))
        if len(rows) != settings["n"] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Incomplete or duplicate {task} inputs: {len(rows)}")
        freeze(root / f"{task}.json", rows)
        print(task, len(rows), digest(rows), flush=True)
