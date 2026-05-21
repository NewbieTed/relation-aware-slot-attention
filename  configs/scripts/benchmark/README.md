# Benchmark Scripts

This directory contains wrappers for generating images and scoring them with
T2I-CompBench. The current project result is measured mainly with the spatial
benchmark.

## Scripts

- `run_flux_relation_t2icompbench.sh`: generate GNN-OSCR-conditioned FLUX images
  and score them with T2I-CompBench.
- `run_flux_vanilla_t2icompbench.sh`: generate baseline FLUX images and score them.
- `summarize_t2i_repeats.sh`: summarize repeated benchmark runs.
- `summarize_t2i_overall.sh`: summarize overall benchmark outputs.
- `t2i_compbench_overrides/`: patched evaluator files used to keep local scoring
  compatible with our environment.

## Example

```bash
cd /local1/cse_481_m_l/relation-aware-slot-attention

CONFIG_FILE=configs/flux/eval_official_seethrough3d_spatial_20p_bf16_512.yaml \
bash scripts/benchmark/run_flux_relation_t2icompbench.sh
```

Avoid running two T2I-CompBench evaluations at the same time against the same
checkout. Some intermediate files inside the benchmark repo can overwrite each
other even when output directories differ.
