"""Repeat historical math and isolated code checks for the selected paired-task arms."""

import argparse
import subprocess
from pathlib import Path

from .capability import read, save, score_task, sha
from .paired_experiment import generate, immutable, load_model
from .capability_data import digest


def run(config):
    root = Path(config["output_dir"])
    selection = read(root / "selection.json")["arms"]
    tasks = [
        row
        for row in read(config["regression_suite"])["tasks"]
        if row["category"] in ("math", "code")
    ]
    if [sum(t["category"] == kind for t in tasks) for kind in ("math", "code")] != [
        30,
        25,
    ]:
        raise ValueError("Expected the frozen 30 math and 25 code tasks")
    subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], check=True)
    image = subprocess.check_output(
        ["docker", "image", "inspect", "python:3.11-slim", "--format", "{{.Id}}"],
        text=True,
    ).strip()
    for name, adapter in selection.items():
        folder = root / "regression" / name
        immutable(
            folder / "protocol.json",
            dict(
                image=image,
                tasks=digest(tasks),
                adapter=sha(Path(adapter) / "adapter_model.safetensors"),
                budget=256,
                dtype="bfloat16",
                batch=4,
            ),
        )
        if not (folder / "responses.json").exists():
            model, tok = load_model(config, adapter)
            rows = []
            for index in range(0, len(tasks), 4):
                batch = tasks[index : index + 4]
                prompts = [
                    dict(
                        t,
                        prompt=t["prompt"]
                        + (
                            "\nAt the end, write The answer is: followed by your final numeric answer."
                            if t["category"] == "math"
                            else ""
                        ),
                    )
                    for t in batch
                ]
                answers = generate(model, tok, prompts, budget=256)
                rows.extend(
                    dict(
                        task_id=t["id"],
                        stage=name,
                        response=a["response"],
                        finish_reason="length" if a["truncated"] else "eos",
                    )
                    for t, a in zip(batch, answers)
                )
            save(folder / "responses.json", rows)
            del model
        responses = read(folder / "responses.json")
        if [r["task_id"] for r in responses] != [t["id"] for t in tasks]:
            raise ValueError("Regression responses do not match the frozen task order")
        scored = [
            score_task(t, r, execute_code=True)
            for t, r in zip(tasks, responses)
        ]
        if len(scored) != 55 or any(r["correct"] is None for r in scored):
            raise RuntimeError("Incomplete regression scores")
        save(folder / "scores.json", scored)
        print(
            name,
            {
                k: sum(r["correct"] for r in scored if r["category"] == k)
                for k in ("math", "code")
            },
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    run(read(parser.parse_args().config))
