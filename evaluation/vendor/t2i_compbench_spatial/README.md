This directory vendors the minimal T2I-CompBench spatial-evaluation stack that
we currently use locally.

Included scope:

- `unidet/2D_spatial_eval.py`
- `unidet/3D_spatial_eval.py`
- the UniDet object-detection and depth dependencies those scripts import
- the prompt-independent feature files and config files the evaluators expect

Why this snapshot exists:

- keeps evaluation self-contained inside this repo
- avoids depending on a separate patched `T2I-CompBench/` checkout
- preserves the Mac/CPU compatibility fixes we needed for local runs

Local compatibility notes:

- object-detection setup falls back to CPU when CUDA is unavailable
- 2D spatial evaluation uses `batch_size=1` and `num_workers=0` for local CPU stability
- filename parsing in the dataset loader uses the rightmost underscore suffix

Weights are still external and are expected under:

- `unidet/experts/expert_weights/Unified_learned_OCIM_RS200_6x+2x.pth`
- `unidet/experts/expert_weights/dpt_hybrid-midas-501f0c75.pt`

They are intentionally gitignored. After a fresh clone, download them with:

```bash
./scripts/benchmark/setup_t2i_compbench_weights.sh
```

Prompt files live outside this vendor snapshot under:

- `evaluation/benchmarks/t2i_compbench/`
