# Evaluation Baseline

This directory contains the first evaluation slice for baseline image generation
and benchmark scoring.

Current scope:

- vanilla Stable Diffusion generation for `sd14`, `sd15`, and `sd21`
- prompt-file generation aligned with T2I-CompBench naming/layout
- vendored T2I-CompBench spatial evaluation components
- official prompt files for the 2D and 3D spatial validation sets

Important paths:

- generator: `evaluation/generate.py`
- benchmark wrapper: `evaluation/t2i_compbench.py`
- official 2D prompts: `evaluation/benchmarks/t2i_compbench/spatial_val.txt`
- official 3D prompts: `evaluation/benchmarks/t2i_compbench/3d_spatial_val.txt`
- vendored evaluator root: `evaluation/vendor/t2i_compbench_spatial/unidet`

## Install

```bash
python3 -m pip install -e ".[eval]"
```

Then download the vendored spatial benchmark weights:

```bash
./scripts/benchmark/setup_t2i_compbench_weights.sh
```

For local T2I-CompBench spatial evaluation on this Mac, we currently use the
separate Python 3.10 environment at:

```bash
/Users/newbieted/workspace/relation-aware-slot-attention/.venv-t2i310sys/bin/python
```

## Example Generation

Dry run on the official 2D spatial validation prompts:

```bash
python3 -m evaluation.generate \
  --model sd15 \
  --prompts-file evaluation/benchmarks/t2i_compbench/spatial_val.txt \
  --output-dir outputs/eval/sd15_t2i_compbench_spatial_val_dryrun \
  --num-images-per-prompt 1 \
  --limit-prompts 20 \
  --start-index 0 \
  --device mps
```

The runner writes:

- generated images to `samples/<prompt>_<global_index>.png`
- run settings to `run_config.json`

## Spatial Scoring

2D spatial:

```bash
python3 -m evaluation.t2i_compbench \
  --generated-dir outputs/eval/sd15_t2i_compbench_spatial_val_dryrun \
  --prompt-file evaluation/benchmarks/t2i_compbench/spatial_val.txt \
  --python-bin /Users/newbieted/workspace/relation-aware-slot-attention/.venv-t2i310sys/bin/python
```

3D spatial:

```bash
python3 -m evaluation.t2i_compbench \
  --benchmark 3d_spatial \
  --generated-dir outputs/eval/sd15_t2i_compbench_3d_spatial_val_dryrun \
  --prompt-file evaluation/benchmarks/t2i_compbench/3d_spatial_val.txt \
  --python-bin /Users/newbieted/workspace/relation-aware-slot-attention/.venv-t2i310sys/bin/python
```

By default, the wrapper now uses the vendored evaluator root inside this repo, so
an external `T2I-CompBench/` checkout is no longer required for 2D/3D spatial
evaluation.

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
MODEL=sd21 DEVICE=cpu LIMIT_PROMPTS=10 ./scripts/benchmark/run_t2i_spatial_dryrun.sh
```
