"""Collect measured CPU traces, then render static figures without retraining.

Run collect from the repository root, or supply --toy-data explicitly.
Matplotlib is imported only when rendering; install with pip install '.[viz]'.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
from .evaluation import score_predictions

FIGURES = ('lora-learning', 'lora-merge', 'decoding', 'toy-evaluation', 'overview')
COLORS = ['#64748b', '#2563eb', '#0891b2', '#7c3aed', '#ea580c']


def write_json(path, report):
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def collect(data_dir, toy_data):
    from .demo import run_cpu_demo
    raw = toy_data.read_bytes()
    rows = [json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) or
                       not all(isinstance(row.get(k), str) for k in ('prediction', 'reference')) for row in rows):
        raise ValueError('Toy data must contain prediction/reference string records')
    evaluation = score_predictions([r['prediction'] for r in rows], [r['reference'] for r in rows])
    demo = run_cpu_demo(include_trace=True)
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False)
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True, check=False)
        commit = revision.stdout.strip() if revision.returncode == 0 else None
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
    except OSError:
        commit, dirty = None, None
    manifest = dict(schema_version=1, source_revision=commit, source_worktree_dirty=dirty,
                    toy_data=dict(path=toy_data.as_posix(), sha256=hashlib.sha256(raw).hexdigest()),
                    reports=['cpu-demo.json', 'toy-evaluation.json'],
                    purpose='CPU implementation validation: synthetic training and toy examples')
    validate_reports(demo, evaluation)
    data_dir.mkdir(parents=True, exist_ok=True)
    write_json(data_dir / 'cpu-demo.json', demo)
    write_json(data_dir / 'toy-evaluation.json', evaluation)
    write_json(data_dir / 'manifest.json', manifest)


def _finite(value):
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)
    elif isinstance(value, (float, int)) and not math.isfinite(value):
        raise ValueError('Reports must contain only finite numbers')


def validate_reports(demo, evaluation):
    """Reject incomplete/inconsistent evidence rather than synthesizing plot data."""
    try:
        _finite(demo)
        _finite(evaluation)
        trace = demo['trace']
        history = trace['loss_history']
        steps = trace['training']['steps']
        if steps != 80 or [r['step'] for r in history] != list(range(81)):
            raise ValueError('Expected a measured loss trajectory for steps 0 through 80')
        if any(r['mse'] <= 0 for r in history):
            raise ValueError('Logarithmic loss plot requires positive MSE values')
        for actual, expected in [(history[0]['mse'], demo['lora_synthetic_loss_before']),
                                 (history[-1]['mse'], demo['lora_synthetic_loss_after'])]:
            if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-12):
                raise ValueError('Loss trajectory endpoints disagree with the summary')
        before, after = [torch.tensor(trace['merge_outputs'][key], dtype=torch.float64) for key in ('before', 'after')]
        if before.ndim != 2 or before.numel() == 0 or before.shape != after.shape:
            raise ValueError('Merge output arrays must have matching nonempty 2D shapes')
        error = (before - after).abs().max().item()
        if not math.isclose(error, demo['lora_merge_max_error'], rel_tol=1e-6, abs_tol=1e-12):
            raise ValueError('Merge error disagrees with the measured outputs')
        distributions = [trace['original_distribution'], *demo['sampling_distributions'].values()]
        if set(demo['sampling_distributions']) != {'greedy', 'topk', 'topp', 'minp'}:
            raise ValueError('Expected four decoding strategies')
        for values in distributions:
            if len(values) != 3 or min(values) < 0 or not math.isclose(sum(values), 1, abs_tol=1e-6):
                raise ValueError('Each distribution must contain three nonnegative probabilities summing to one')
        settings = trace['sampling_settings']
        from .decoding import DecodingSampler
        logits = torch.tensor(trace['sampling_logits'], dtype=torch.float32)
        if logits.shape != (3,) or not torch.allclose(torch.tensor(trace['original_distribution']),
                                                    torch.softmax(logits, 0), atol=1e-6, rtol=1e-5):
            raise ValueError('Original probabilities disagree with the recorded logits')
        for strategy, values in demo['sampling_distributions'].items():
            expected = DecodingSampler().distribution(logits, strategy=strategy, **settings)
            if not torch.allclose(torch.tensor(values), expected, atol=1e-6, rtol=1e-5):
                raise ValueError('Recorded sampling probabilities disagree with their settings')
        if evaluation['total'] == 0:
            raise ValueError('Toy evaluation must be nonempty')
        rescored = score_predictions([r['prediction'] for r in evaluation['records']],
                                     [r['reference'] for r in evaluation['records']], evaluation['mode'])
        if rescored != evaluation:
            raise ValueError('Toy evaluation summary or records disagree with rescoring')
        return demo, evaluation
    except (KeyError, TypeError, RuntimeError, IndexError) as exc:
        raise ValueError(f'Missing or malformed visualization data: {exc}') from exc


def load_reports(data_dir):
    demo = json.loads((data_dir / 'cpu-demo.json').read_text(encoding='utf-8'))
    evaluation = json.loads((data_dir / 'toy-evaluation.json').read_text(encoding='utf-8'))
    return validate_reports(demo, evaluation)


def draw_loss(ax, demo):
    history = demo['trace']['loss_history']
    ax.plot([r['step'] for r in history], [r['mse'] for r in history], color=COLORS[1], lw=2)
    ax.scatter([0, 80], [history[0]['mse'], history[-1]['mse']], color=COLORS[1], s=24, zorder=3)
    ax.set(yscale='log', xlabel='Optimizer step', ylabel='MSE (log scale)', title='LoRA learning | Synthetic regression')
    ax.text(.97, .96, f"Step 0: {history[0]['mse']:.3g}\nStep 80: {history[-1]['mse']:.3g}",
            ha='right', va='top', transform=ax.transAxes, fontsize=9)
    ax.grid(axis='y', alpha=.2)


def draw_merge(axes, demo):
    before, after = [torch.tensor(demo['trace']['merge_outputs'][key], dtype=torch.float64).flatten()
                     for key in ('before', 'after')]
    error = (before - after).abs()
    low, high = min(before.min(), after.min()).item(), max(before.max(), after.max()).item()
    axes[0].scatter(before, after, s=10, alpha=.5, color=COLORS[2])
    axes[0].plot([low, high], [low, high], '--', color=COLORS[0], lw=1, label='y = x')
    axes[0].set(title='LoRA merge | Outputs', xlabel='Before merge', ylabel='After merge')
    axes[0].legend(fontsize=8)
    axes[1].plot(error.tolist(), color=COLORS[2], lw=1)
    axes[1].set(title='LoRA merge | Absolute error', xlabel='Output element', ylabel='Absolute difference')
    axes[1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[1].text(.97, .94, f'Max: {error.max().item():.2e}', ha='right', va='top', transform=axes[1].transAxes, fontsize=8)
    axes[1].set_ylim(bottom=0, top=max(error.max().item() * 1.4, 1e-8))


def draw_decoding(ax, demo):
    trace = demo['trace']
    settings = trace['sampling_settings']
    distributions = [trace['original_distribution'], *[demo['sampling_distributions'][s] for s in ('greedy', 'topk', 'topp', 'minp')]]
    labels = ['Original', 'Greedy', 'Top-K', 'Top-P', 'Min-P']
    width = .15
    for j, (label, values) in enumerate(zip(labels, distributions)):
        ax.bar([i + (j - 2) * width for i in range(3)], values, width=width, label=label, color=COLORS[j])
    ax.set(xticks=[0, 1, 2], xticklabels=['Token A', 'Token B', 'Token C'], ylim=(0, 1.25), ylabel='Probability',
           title=f"Decoding | T={settings['temperature']:g}, k={settings['top_k']}, p={settings['top_p']}, min_p={settings['min_p']}")
    ax.legend(ncol=3, loc='upper right', fontsize=8, frameon=False)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)


def draw_evaluation(ax, evaluation):
    counts = [sum(r['status'] == status for r in evaluation['records']) for status in ('correct', 'incorrect', 'invalid_output')]
    bars = ax.barh(['Correct', 'Incorrect', 'Invalid output'], counts, color=[COLORS[2], COLORS[4], COLORS[0]])
    ax.bar_label(bars, padding=4)
    ax.set(xlim=(0, max(counts) + 1), xlabel='Number of examples', title=f"Toy examples | n={evaluation['total']}")
    ax.set_xticks(range(max(counts) + 2))
    ax.invert_yaxis()


def render(data_dir, output_dir):
    demo, evaluation = load_reports(data_dir)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ValueError("Rendering requires Matplotlib: pip install '.[viz]'") from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
                         'figure.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False}):
        for name in FIGURES:
            fig = plt.figure(figsize=(12, 8) if name == 'overview' else (8, 4.5), layout='constrained')
            if name == 'overview':
                grid = fig.add_gridspec(2, 2, hspace=.18, wspace=.08)
                draw_loss(fig.add_subplot(grid[0, 0]), demo)
                draw_decoding(fig.add_subplot(grid[0, 1]), demo)
                merge_grid = grid[1, 0].subgridspec(1, 2, wspace=.15)
                draw_merge([fig.add_subplot(merge_grid[0, i]) for i in range(2)], demo)
                draw_evaluation(fig.add_subplot(grid[1, 1]), evaluation)
                fig.suptitle('NLP components | CPU implementation validation\nSynthetic training and toy examples; no language-model benchmark claims', fontsize=14)
            elif name == 'lora-merge':
                draw_merge(fig.subplots(1, 2), demo)
            else:
                ax = fig.subplots()
                if name == 'lora-learning':
                    draw_loss(ax, demo)
                elif name == 'decoding':
                    draw_decoding(ax, demo)
                else:
                    draw_evaluation(ax, evaluation)
            if name != 'overview':
                fig.suptitle('CPU demonstration | Seed 42 | Synthetic / toy data', fontsize=10, color=COLORS[0])
            try:
                fig.savefig(output_dir / f'{name}.png', dpi=180)
                fig.savefig(output_dir / f'{name}.svg')
            finally:
                plt.close(fig)
    return [output_dir / f'{name}.{ext}' for name in FIGURES for ext in ('png', 'svg')]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    collector = commands.add_parser('collect', help='Run the original CPU demo and score the toy fixture')
    collector.add_argument('--data-dir', type=Path, default=Path('docs/validation/visualization'))
    collector.add_argument('--toy-data', type=Path, default=Path('examples/toy_predictions.jsonl'))
    renderer = commands.add_parser('render', help='Render saved reports only; no training')
    renderer.add_argument('--data-dir', type=Path, default=Path('docs/validation/visualization'))
    renderer.add_argument('--output-dir', type=Path, default=Path('docs/figures'))
    args = parser.parse_args(argv)
    try:
        if args.command == 'collect':
            collect(args.data_dir, args.toy_data)
            print(f'Collected measured reports in {args.data_dir}')
        else:
            for path in render(args.data_dir, args.output_dir):
                print(path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
