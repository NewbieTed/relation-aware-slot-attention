# Benchmark Setup Scripts

This directory prepares optional benchmark dependencies. These are separated
from the core setup because T2I-CompBench, GenEval, and VISOR have heavier or
more fragile dependencies than the project itself.

## Scripts

- `setup_t2i_compbench_checkout.sh`: clones T2I-CompBench.
- `setup_t2i_compbench_env.sh`: creates the benchmark-compatible Python env.
- `setup_t2i_compbench_weights.sh`: downloads model weights expected by T2I-CompBench.
- `setup_geneval_checkout.sh`, `setup_geneval_env.sh`, `setup_geneval_models.sh`: prepare GenEval.
- `setup_visor_checkout.sh`: prepares VISOR.

## Example

```bash
cd /local1/cse_481_m_l/relation-aware-slot-attention

bash scripts/setup/benchmark/setup_t2i_compbench_checkout.sh
PYTHON_BIN=/homes/iws/yixuan19/miniforge3/envs/t2i310/bin/python \
bash scripts/setup/benchmark/setup_t2i_compbench_env.sh
bash scripts/setup/benchmark/setup_t2i_compbench_weights.sh
```
