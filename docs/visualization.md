# Static result figures / 静态成果图

The figures show CPU implementation validation on synthetic training and toy examples. They are not pretrained-language-model benchmark results. 图表用于解释实现行为，不代表真实语言模型性能。

## Reproduce / 重新生成

From the repository root, with the project's CPU PyTorch already installed:

```sh
python -m pip install -e '.[viz]'
python -m nlp_course.visualization collect --data-dir docs/validation/visualization
python -m nlp_course.visualization render --data-dir docs/validation/visualization --output-dir docs/figures
```

Collection reruns the original demonstration with seed 42 and records actual data. Rendering reads the saved JSON only and does not retrain. To change styling, rerun only `render`. Both commands accept custom directories; `collect --toy-data PATH` selects a JSONL fixture with prediction/reference strings. The default fixture is `examples/toy_predictions.jsonl`, so run the default collection command from the repository root.

`run_cpu_demo(include_trace=False)` is available from `nlp_course.demo`. Its default return fields match the original example. With `include_trace=True`, the additional `trace` object records training configuration/environment, measured loss trajectory, paired merge outputs, logits and sampling settings. `examples/cpu_demo.py` continues to print only the original JSON report.

采集命令重新执行示例，绘图命令只读取报告。新增报告位于独立目录，历史验证文件保持原样。使用自定义数据目录可以保留不同环境的运行结果。

## Data and figures / 数据与图表

| Figure | Data source | Meaning |
| --- | --- | --- |
| [Learning curve](figures/lora-learning.svg) | [CPU trace](validation/visualization/cpu-demo.json): `trace.loss_history` | 81 measured MSE values, optimizer steps 0–80; logarithmic y-axis, no smoothing or interpolation |
| [Merge consistency](figures/lora-merge.svg) | CPU trace: `trace.merge_outputs` | Same 64×3 outputs before/after inference merge; identity reference and absolute error per element |
| [Decoding](figures/decoding.svg) | CPU trace: original and filtered distributions | Three tokens; temperature 1, k=2, p=0.8, relative min_p=0.5; float rounding at a threshold may change inclusion |
| [Toy outcomes](figures/toy-evaluation.svg) | [Four-example report](validation/visualization/toy-evaluation.json) | Count statuses from records: correct, incorrect, invalid output; excludes the CPU demo's separate two-example report |
| [Overview](figures/overview.svg) | Same drawing functions and reports | Compact 2×2 arrangement; merge panel has two subplots |

All figures have SVG and PNG versions. Individual figures are 8×4.5 inches / 1440×810 pixels at 180 DPI; the overview is 12×8 inches / 2160×1440 pixels. English figure text and the bundled DejaVu Sans font avoid platform-specific CJK fonts. Both READMEs include captions and links to source data.

The [manifest](validation/visualization/manifest.json) identifies the input fixture by SHA-256 and records the Git revision and dirty-working-tree flag when Git metadata is available. A dirty flag means the reports were collected before committing local changes; the revision alone is not an exact snapshot of those working-tree changes. Environment and training settings are recorded inside the CPU trace.

## Validation / 验证

The renderer rejects incomplete trajectories, nonfinite data, invalid probability distributions, mismatched output arrays and summaries inconsistent with the data. It recomputes decoding distributions from the recorded logits/settings and rescores evaluation records using the existing implementation. It never infers missing intermediate loss values.

Tests cover compatibility, source/summary agreement, missing/malformed reports, lazy plotting imports, file formats and exact expected PNG dimensions. Rendering tests run when the `viz` extra is installed; the core test matrix does not require Matplotlib. Pixel-identical images are not required across platforms.

The additional Ubuntu / Python 3.11 CI job installs `.[viz]`, runs tests, collects reports, renders all ten files, and uploads reports plus images as `static-visualization-py311`. It writes under ignored `outputs/` and does not overwrite committed figures. Inspect the generated PNGs visually before updating the README's published images.
