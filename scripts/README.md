# Scripts

This directory contains shell wrappers for setup, training, and benchmark
workflows. The scripts are intentionally thin: they set environment variables,
activate the expected virtualenv, and call the Python modules in `training/`,
`evaluation/`, or `scop_depth/`.

## Subdirectories

- `setup/`: create Python environments and clone external benchmark/model repos.
- `train/`: launch GNN pretraining.
- `benchmark/`: run T2I-CompBench and summarize repeat results.

## Common Remote Pattern

```bash
cd /local1/cse_481_m_l/relation-aware-slot-attention
source .venv-flux/bin/activate
export HF_HOME=/local1/cse_481_m_l/relation-aware-slot-attention/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
```

Use `logs/` for long-running `nohup` outputs.
