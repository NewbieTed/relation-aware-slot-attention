# Evaluation Baseline

This directory contains the first evaluation slice for baseline image generation
and benchmark scoring.

Current scope:

- vanilla Stable Diffusion generation for `sd14`, `sd15`, and `sd21`
- prompt-file generation aligned with benchmark-friendly naming/layout
- wrappers for T2I-CompBench categories and overall aggregation
- a wrapper for GenEval evaluation
- a scoring wrapper for VISOR once object-detection results are available
- a lightweight local override layer for the patched benchmark files we need

Important paths:

- generator: `evaluation/generate.py`
- T2I-CompBench wrapper: `evaluation/t2i_compbench.py`
- T2I-CompBench overall aggregator: `evaluation/t2i_compbench_overall.py`
- GenEval wrapper: `evaluation/geneval.py`
- VISOR scoring wrapper: `evaluation/visor.py`
- full onboarding bootstrap: `scripts/setup/bootstrap_all.sh`
- environment bootstrap: `scripts/setup/bootstrap_env.sh`
- benchmark-only bootstrap: `scripts/setup/bootstrap_benchmark.sh`
- benchmark checkout helper: `scripts/setup/benchmark/setup_t2i_compbench_checkout.sh`
- benchmark weights helper: `scripts/setup/benchmark/setup_t2i_compbench_weights.sh`
- T2I extra Python env helper: `scripts/setup/benchmark/setup_t2i_compbench_env.sh`
- GenEval checkout helper: `scripts/setup/benchmark/setup_geneval_checkout.sh`
- GenEval extra Python env helper: `scripts/setup/benchmark/setup_geneval_env.sh`
- GenEval model helper: `scripts/setup/benchmark/setup_geneval_models.sh`
- VISOR checkout helper: `scripts/setup/benchmark/setup_visor_checkout.sh`

## Install

Simplest onboarding path:

```bash
./scripts/setup/bootstrap_all.sh
```

This will:

- create `.venv`
- install the repo in editable mode with evaluation extras
- install the extra benchmark Python packages we rely on
- download the `en_core_web_sm` spaCy model
- attempt Detectron2 installation on Linux
- clone T2I-CompBench into `external/T2I-CompBench`
- clone GenEval into `external/geneval`
- clone VISOR into `external/VISOR`
- apply our compatibility overrides
- download the benchmark weights

If you want to run the setup in pieces instead, use:

```bash
./scripts/setup/bootstrap_env.sh
./scripts/setup/bootstrap_benchmark.sh
```

GenEval's detector weights are optional and can be downloaded separately:

```bash
./scripts/setup/benchmark/setup_geneval_models.sh
```

Benchmark-specific Python extras can also be installed separately:

```bash
./scripts/setup/benchmark/setup_t2i_compbench_env.sh
./scripts/setup/benchmark/setup_geneval_env.sh
```

`setup_t2i_compbench_env.sh` creates a dedicated `.venv-t2i` by default so the
older T2I-CompBench dependencies do not conflict with the repo's main `.venv`.
That dedicated environment also pins older benchmark-compatible versions such as
`transformers==4.30.2`, `timm==0.4.12`, and `Pillow==9.5.0`, which should not be
installed into the main repo environment.

For GenEval, the script installs the lightweight Python extras by default and
skips the legacy MMDetection/MMCV detector stack unless you explicitly opt in:

```bash
INSTALL_MMDET_STACK=1 ./scripts/setup/benchmark/setup_geneval_env.sh
```

That detector stack was designed around much older PyTorch/CUDA combinations,
so it may require a more specialized environment than the repo's main `.venv`.

For local T2I-CompBench evaluation, it is often easiest to use a separate Python
3.10 environment. The bootstrap script uses your current `python3` by default,
so if you need a specific interpreter you can override it:

```bash
PYTHON_BIN=python3.10 ./scripts/setup/bootstrap_all.sh
```

To build the benchmark-only compatibility environment explicitly:

```bash
PYTHON_BIN=python3.10 ./scripts/setup/benchmark/setup_t2i_compbench_env.sh
```

If an older bootstrap created `scripts/.venv`, delete that stale environment
before rerunning the new setup flow.

The official prompt files now come from the external benchmark checkout:

- `external/T2I-CompBench/examples/dataset/spatial_val.txt`
- `external/T2I-CompBench/examples/dataset/3d_spatial_val.txt`
- `external/geneval/prompts/generation_prompts.txt`
- `external/geneval/prompts/evaluation_metadata.jsonl`

## Example Generation

Dry run on the official 2D spatial validation prompts:

```bash
python3 -m evaluation.generate \
  --model sd15 \
  --prompts-file external/T2I-CompBench/examples/dataset/spatial_val.txt \
  --output-dir outputs/eval/sd15_t2i_compbench_spatial_val_dryrun \
  --num-images-per-prompt 1 \
  --limit-prompts 20 \
  --start-index 0 \
  --device auto
```

The runner writes:

- generated images to `samples/<prompt>_<global_index>.png`
- run settings to `run_config.json`

## Spatial Scoring

2D spatial:

```bash
python3 -m evaluation.t2i_compbench \
  --t2i-compbench-root external/T2I-CompBench \
  --generated-dir outputs/eval/sd15_t2i_compbench_spatial_val_dryrun \
  --prompt-file external/T2I-CompBench/examples/dataset/spatial_val.txt \
  --python-bin python3
```

3D spatial:

```bash
python3 -m evaluation.t2i_compbench \
  --benchmark 3d_spatial \
  --t2i-compbench-root external/T2I-CompBench \
  --generated-dir outputs/eval/sd15_t2i_compbench_3d_spatial_val_dryrun \
  --prompt-file external/T2I-CompBench/examples/dataset/3d_spatial_val.txt \
  --python-bin python3
```

The wrapper accepts either the benchmark repo root or the `UniDet_eval`
subdirectory via `--t2i-compbench-root`.

## T2I-CompBench Categories

Supported single-category benchmarks:

- `color`
- `shape`
- `texture`
- `spatial`
- `non_spatial`
- `complex`
- `numeracy`
- `3d_spatial`

Example dry run for a non-spatial category:

```bash
CATEGORY=non_spatial LIMIT_PROMPTS=20 ./scripts/benchmark/run_t2i_category_dryrun.sh
```

Example full run for numeracy:

```bash
CATEGORY=numeracy NUM_IMAGES_PER_PROMPT=5 DEVICE=cuda ./scripts/benchmark/run_t2i_category_full.sh
```

The original T2I-CompBench "overall" score is an aggregate over:

- `color`
- `shape`
- `texture`
- `spatial`
- `non_spatial`
- `complex`

After running those category evaluations into a common root layout, summarize them with:

```bash
python3 -m evaluation.t2i_compbench_overall \
  --root-dir /path/to/t2i_outputs
```

Or use the helper script:

```bash
ROOT_OUTPUT_DIR=/path/to/t2i_outputs ./scripts/benchmark/summarize_t2i_overall.sh
```

This intentionally keeps `numeracy` and `3d_spatial` separate, since those belong to the extended T2I-CompBench++ style evaluation surface rather than the original overall score.

## GenEval

The GenEval wrapper expects:

- generated images from `external/geneval/prompts/generation_prompts.txt`
- the matching metadata file `external/geneval/prompts/evaluation_metadata.jsonl`
- downloaded detector weights, usually via `./scripts/setup/benchmark/setup_geneval_models.sh`

Dry run:

```bash
./scripts/benchmark/run_geneval_dryrun.sh
```

Full run:

```bash
NUM_IMAGES_PER_PROMPT=4 DEVICE=cuda ./scripts/benchmark/run_geneval_full.sh
```

The wrapper writes:

- `geneval_results.jsonl`
- `geneval_eval.json`

inside the chosen output directory.

## VISOR

The VISOR wrapper currently handles the scoring stage once you already have object-detection results in VISOR's JSON format.

Example:

```bash
python3 -m evaluation.visor \
  --visor-root external/VISOR \
  --results-json /path/to/results_model_text_spatial_rel_phrases_owlvit_0.1.json
```

This writes a summary JSON next to the provided results file with:

- `OA`
- `VISOR_cond`
- `VISOR_uncond`
- `VISOR_1`
- `VISOR_2`
- `VISOR_3`
- `VISOR_4`

## Shell Scripts

For convenience, the common benchmark runs are wrapped in:

- `scripts/benchmark/run_t2i_category_dryrun.sh`
- `scripts/benchmark/run_t2i_category_full.sh`
- `scripts/benchmark/run_t2i_spatial_dryrun.sh`
- `scripts/benchmark/run_t2i_spatial_full.sh`
- `scripts/benchmark/run_t2i_3d_spatial_dryrun.sh`
- `scripts/benchmark/run_t2i_3d_spatial_full.sh`
- `scripts/benchmark/run_geneval_dryrun.sh`
- `scripts/benchmark/run_geneval_full.sh`
- `scripts/benchmark/summarize_t2i_overall.sh`

The generic T2I category scripts are the primary path. The older `run_t2i_spatial_*`
and `run_t2i_3d_spatial_*` scripts are now thin compatibility aliases that forward
to the generic category runner with `CATEGORY=spatial` or `CATEGORY=3d_spatial`.

Example:

```bash
./scripts/benchmark/run_t2i_spatial_dryrun.sh
```

These scripts support simple environment-variable overrides, for example:

```bash
MODEL=sd21 DEVICE=cuda LIMIT_PROMPTS=10 ./scripts/benchmark/run_t2i_spatial_dryrun.sh
```

If `.venv/bin/python` exists, the run scripts will use it automatically.
