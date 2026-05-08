# Configs

This directory stores YAML configs for repeatable research runs. Config files
are preferred over long command-line flag lists so that training, generation,
and benchmark settings can be reviewed and reused.

## Layout

- `flux/`: configs for the current FLUX + GNN-generated OSCR workflow.

## Usage

Most Python entrypoints accept `--config` and read the section they need from
the YAML file. The benchmark shell wrappers use `CONFIG_FILE`.

```bash
CONFIG_FILE=/local1/cse_481_m_l/relation-aware-slot-attention/configs/flux/eval_official_seethrough3d_spatial_1p_bf16_512.yaml \
bash scripts/benchmark/run_flux_relation_t2icompbench.sh
```

Config mode is intentionally strict: when `--config` is used, extra CLI
overrides are rejected. Edit the YAML instead so the run remains reproducible.
