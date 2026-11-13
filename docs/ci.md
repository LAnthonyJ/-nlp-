# CI coverage and validation reports

The [CPU tests workflow](../.github/workflows/tests.yml) checks the following five combinations:

| Runner | Python |
| --- | --- |
| Ubuntu | 3.10 |
| Ubuntu | 3.11 |
| Ubuntu | 3.12 |
| Ubuntu | 3.13 |
| Windows | 3.13 |

Linux jobs cover the selected Python versions, including the declared minimum of 3.10. Windows exercises the development platform. This matrix does not assert support for every operating system, every PyTorch release, or future Python versions. CI uses the pinned CPU dependency from `requirements-cpu.txt`.

## What each job verifies

1. Install CPU-only PyTorch, build/install the project as a regular package, and check dependency consistency with `pip check`.
2. Import all four modules from outside the checkout, proving the installed package is usable without an editable source-path setup.
3. Run the complete unittest regression suite, including source-snapshot hashes.
4. Run the CPU demo and JSONL evaluation commands.
5. Check Transformer output shape, decreasing synthetic LoRA loss, merge error below `1e-5`, and expected toy-evaluation counts.

These checks validate software behavior on small examples; the toy accuracy is not a model benchmark.

## Downloading the evidence

Open [CPU tests runs](https://github.com/LAnthonyJ/-nlp-/actions/workflows/tests.yml), select a run, and inspect the job summaries or the **Artifacts** section. Artifacts are named `cpu-ubuntu-latest-py3.10`, etc., and retained for 14 days.

Each successful job provides:

- `environment.json`: Python, OS, PyTorch and CPU-build information.
- `dependencies.txt`: installed dependency versions.
- `tests.log`: verbose regression results.
- `cpu-demo.json`: synthetic optimization and sampling results.
- `toy-report.json`: per-example evaluation results.
- `evaluation.log`: evaluation console output.

Artifact upload is attempted even after a prior step fails; only files produced before the failure will be present. Install failures may occur before any report exists. The job itself still fails when tests or examples fail; pipes use GitHub's explicit Bash shell with `pipefail`. Bash is used on both hosted operating systems for consistent logging.

## Runtime controls

- Trigger on pushes to `main`, pull requests, or manual dispatch.
- Cancel superseded runs for the same workflow/ref.
- Keep other matrix jobs running if one fails, so compatibility failures are visible.
- Limit each job to 20 minutes.
- Cache pip downloads by Python/runner and dependency-file hashes.
- Use read-only repository permissions and avoid persisting checkout credentials.

Action configuration follows the official [setup-python](https://github.com/actions/setup-python), [upload-artifact](https://github.com/actions/upload-artifact) and [checkout](https://github.com/actions/checkout) documentation. This workflow uses their v7 major references on GitHub-hosted runners.
