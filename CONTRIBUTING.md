# Contributing to TrustTour

Thanks for your interest in improving TrustTour! Contributions of all kinds are welcome — bug fixes, new baselines, dataset loaders, documentation, and performance improvements.

## Getting Started

```bash
git clone https://github.com/<your-org>/trusttour.git
cd trusttour
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

Please make sure all tests pass (and add new tests for new functionality) before opening a pull request.

## Code Style

- Follow PEP 8; format with `black` (line length 100 is fine).
- Type hints are encouraged, especially in `models/`, `data/`, `rl/`, and `game/`.
- Keep modules focused: one architectural component per file (mirrors the paper's modular structure: Trust Assessment, Adaptive Decision, Strategic Optimization).

## Adding a New Baseline

1. Create `models/baselines/<name>.py` exposing a `build_<name>(**kwargs)` factory that returns an `nn.Module` whose `forward(x, return_features=False)` returns `(prob, logit)` or `(prob, logit, features)`.
2. Register it in `models/baselines/registry.py`.
3. Add a config entry to `configs/baselines.yaml`.
4. Add a smoke test to `tests/test_models.py`.

## Adding a New Dataset

1. Add a `DatasetSpec` entry to `data/datasets.py` (`DATASET_SPECS`) describing the expected real/fake subfolder names.
2. Verify with `python data/prepare_data.py --root <path> --dataset <name> --check`.

## Pull Requests

- Keep PRs focused on a single change.
- Include a short description of what changed and why.
- Reference any related issues.

## Reporting Issues

Please include: Python/PyTorch versions, OS, a minimal reproduction script, and the full traceback.
