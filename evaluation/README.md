# Evaluation Baseline

This directory contains the first evaluation slice for baseline image generation
and benchmark scoring.

Current scope:

- vanilla Stable Diffusion generation for `sd14`, `sd15`, and `sd21`
- prompt-file generation aligned with T2I-CompBench naming/layout
- wrapper scripts for T2I-CompBench spatial evaluation
- a lightweight local override layer for the patched benchmark files we need

Important paths:

- generator: `evaluation/generate.py`
- benchmark wrapper: `evaluation/t2i_compbench.py`
- master benchmark bootstrap: `scripts/setup/bootstrap_benchmark.sh`
- benchmark checkout helper: `scripts/setup/benchmark/setup_t2i_compbench_checkout.sh`
- benchmark weights helper: `scripts/setup/benchmark/setup_t2i_compbench_weights.sh`

## Install

```bash
python3 -m pip install -e ".[eval]"
```

Then prepare the external T2I-CompBench checkout plus our local compatibility
overrides:

```bash
./scripts/setup/bootstrap_benchmark.sh
```

For local T2I-CompBench evaluation, it is often easiest to use a separate Python
3.10 environment. If needed, point `--python-bin` at that interpreter.

The official prompt files now come from the external benchmark checkout:

- `external/T2I-CompBench/examples/dataset/spatial_val.txt`
- `external/T2I-CompBench/examples/dataset/3d_spatial_val.txt`

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

## Shell Scripts

For convenience, the common benchmark runs are wrapped in:

- `scripts/benchmark/run_t2i_spatial_dryrun.sh`
- `scripts/benchmark/run_t2i_spatial_full.sh`
- `scripts/benchmark/run_t2i_3d_spatial_dryrun.sh`
- `scripts/benchmark/run_t2i_3d_spatial_full.sh`

Example:

```bash
./scripts/benchmark/run_t2i_spatial_dryrun.sh
```

These scripts support simple environment-variable overrides, for example:

```bash
MODEL=sd21 DEVICE=cuda LIMIT_PROMPTS=10 ./scripts/benchmark/run_t2i_spatial_dryrun.sh
```
