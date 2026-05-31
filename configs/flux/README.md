# FLUX Configs

This directory contains configs for the current FLUX branch. The active path is:
train/debug the GNN, render GNN-predicted 3D cuboids into OSCR conditions, then
run inference with the released SeeThrough3D FLUX LoRA.

## Files

Training configs live directly in `configs/flux/`. Older exploratory configs
are under `configs/flux/trash/` so they remain recoverable without crowding the
active experiment list. Evaluation configs are split by purpose:

- `eval/smoke/`: tiny 3-prompt, 1-image runs for checking that generation and
  scoring still work before starting expensive jobs.
- `eval/full/`: final 512px, 5-images-per-prompt benchmark configs for the
  paper comparison.
- `eval/official/`: longer SeeThrough3D/FLUX benchmark configs used for actual
  reporting.
- `eval/layout/`: direct layout-module evaluation configs. These produce
  `paper_table.md`, `paper_table.csv`, per-sample metrics, and per-relation
  metrics without running FLUX image generation.

- `gnn_pretrain_orig_cvae_nofilm_nofreebits_bs128_3200.yaml`: base-data CVAE
  ablation without freebits.
- `gnn_pretrain_orig_cvae_nofilm_freebits05_bs128_3200.yaml`: base-data CVAE
  with freebits `0.5`.
- `gnn_pretrain_orig_cvae_nofilm_freebits1_bs128_3200.yaml`: base-data CVAE
  with freebits `1.0`.
- `gnn_pretrain_orig_cvae_film1_nofreebits_bs128_3200.yaml`: base-data CVAE
  with FiLM scale `1.0` but no freebits.
- `gnn_pretrain_aug_prompt250_deterministic_t5_bs128_96000.yaml`: deterministic
  GNN trained on the prompt-balanced 250-variant augmented dataset.
- `gnn_pretrain_aug_prompt400_deterministic_t5_bs128_154000.yaml`:
  deterministic GNN trained on the prompt-balanced 400-variant augmented
  dataset.
- `gnn_pretrain_aug_prompt250_gnn_film1_freebits1_nowarmup_bs128_96000.yaml`:
  prompt-balanced 250-variant CVAE with FiLM scale `1.0` and freebits `1.0`.
- `gnn_pretrain_aug_prompt250_gnn_nofilm_freebits1_nowarmup_bs128_96000.yaml`:
  same augmented CVAE without FiLM.
- `gnn_pretrain_aug_prompt400_gnn_film1_freebits1_nowarmup_bs128_154000.yaml`
  and `gnn_pretrain_aug_prompt400_gnn_nofilm_freebits1_nowarmup_bs128_154000.yaml`:
  longer prompt-balanced 400-variant CVAE runs. The latest table currently uses
  `checkpoint-045000` snapshots for these CVAE runs.
- `eval/layout/layout_eval_ready_now.yaml`: current paper-facing layout table
  config. It includes priors, relation heuristic, deterministic GNN baselines,
  base-data CVAE ablations, prompt250 final CVAEs, prompt400 CVAE checkpoints,
  and random jitter.
- `eval/layout/layout_eval_prompt_balanced_checkpoints.yaml`: compact config
  for only the prompt-balanced CVAE checkpoint/final models.
- `eval/official/eval_official_seethrough3d_spatial_1p_bf16_512.yaml`: one-prompt smoke eval.
- `eval/official/eval_official_seethrough3d_spatial_20p_bf16_512.yaml`: short benchmark sanity check.
- `eval/official/eval_official_seethrough3d_spatial_full_bf16_8bit_512.yaml`: full spatial benchmark run.

## Examples

Train a base-data 3D_SLN-aligned triple-GNN CVAE:

```bash
python3 -m training.precompute_graph_label_cache \
  --config configs/flux/gnn_pretrain_orig_cvae_nofilm_freebits1_bs128_3200.yaml

python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_orig_cvae_nofilm_freebits1_bs128_3200.yaml
```

Run the current direct layout evaluation:

```bash
python3 -m evaluation.evaluate_layout_models \
  --config configs/flux/eval/layout/layout_eval_ready_now.yaml
```

Run the one-prompt SeeThrough3D smoke benchmark:

```bash
CONFIG_FILE=configs/flux/eval/official/eval_official_seethrough3d_spatial_1p_bf16_512.yaml \
bash scripts/benchmark/run_flux_relation_t2icompbench.sh
```

On the 24GB tomago GPUs, the working FLUX inference setting is currently
`bf16` plus `8bit` transformer quantization with `512` image and OSCR sizes.
