# Setup Scripts

This directory contains onboarding scripts for local or remote environments.
They install Python dependencies, clone SeeThrough3D, and optionally prepare
benchmark repositories.

## Scripts

- `bootstrap_env.sh`: creates the Python virtual environment and installs this repo.
- `setup_seethrough3d_checkout.sh`: clones the SeeThrough3D repository under `external/`.
- `bootstrap_benchmark.sh`: prepares benchmark checkouts.
- `bootstrap_all.sh`: runs the main setup steps together.

## Example

```bash
cd /local1/cse_481_m_l/relation-aware-slot-attention

PYTHON_BIN=/homes/iws/yixuan19/miniforge3/envs/t2i310/bin/python \
VENV_DIR=.venv-flux \
./scripts/setup/bootstrap_env.sh

./scripts/setup/setup_seethrough3d_checkout.sh
```

FLUX.1-dev and the official SeeThrough3D LoRA are gated Hugging Face downloads,
so run `huggingface-cli login` in the same environment before inference.
