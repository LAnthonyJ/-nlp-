"""Prospective paired macro-accuracy gate for the three-stage experiment."""

from .progression_data import TASKS


def scores(rows):
    result = {}
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction")
    for task in TASKS:
        values = [r["correct"] for r in rows if r["task"] == task]
        if not values or any(type(v) is not bool for v in values):
            raise ValueError(f"Missing or invalid {task} outcomes")
        result[task] = 100 * sum(values) / len(values)
    result["mean"] = sum(result.values()) / len(TASKS)
    return result


def compare(before, after, target, formal=False):
    import numpy as np

    first, second = scores(before), scores(after)
    index = {r["id"]: r for r in before}
    if set(index) != {r["id"] for r in after}:
        raise ValueError("Comparison needs identical IDs")
    by_task = {}
    rng = np.random.default_rng(42)
    bootstrap = np.zeros(50000) if formal else None
    for t in TASKS:
        subset = [r for r in after if r["task"] == t]
        if any(index[r["id"]]["task"] != t for r in subset):
            raise ValueError("Task changed across stages")
        diff = [int(r["correct"]) - int(index[r["id"]]["correct"]) for r in subset]
        n, up, down = len(diff), diff.count(1), diff.count(-1)
        by_task[t] = dict(pp=100 * (up - down) / n, gained=up, lost=down, n=n)
        if formal:
            draws = rng.multinomial(
                n, [up / n, down / n, (n - up - down) / n], len(bootstrap)
            )
            bootstrap += 100 * (draws[:, 0] - draws[:, 1]) / (len(TASKS) * n)
    delta = second["mean"] - first["mean"]
    passed = (
        delta + 1e-10 >= target and min(x["pp"] for x in by_task.values()) >= -2 - 1e-10
    )
    result = dict(pp=delta, tasks=by_task, passed=passed)
    if formal:
        lower, lo, hi = map(float, np.quantile(bootstrap, [0.05 / 6, 0.025, 0.975]))
        result.update(lower=lower, ci95=[lo, hi], passed=passed and lower > 0)
    return result
