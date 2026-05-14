# FLUX Configs

This directory contains configs for the current FLUX branch. The active path is:
train/debug the GNN, render GNN-predicted 3D cuboids into OSCR conditions, then
run inference with the released SeeThrough3D FLUX LoRA.

## Files

- `gnn_pretrain_3dbox_triple_cvae_3dsln.yaml`: active GNN training config. It
  uses FLUX T5 object-label embeddings, 5 graph layers, L1 min/max box
  reconstruction, scene/object CVAE latents, KL weight `0.1`, and a persistent
  label-embedding cache under `outputs/cache`.
- `gnn_pretrain_3dbox_deterministic_clip.yaml`: deterministic 3D box GNN
  recovery config using FLUX CLIP label embeddings. This is the closest config
  to the earlier deterministic GNN path: 2 graph layers, 600 steps, center loss,
  relation loss, and 3D size loss.
- `gnn_pretrain_3dbox_deterministic_t5.yaml`: deterministic 3D box GNN with the
  same losses and schedule as the CLIP version, but using FLUX T5 label
  embeddings so we can compare the text encoder effect directly.
- `eval_official_seethrough3d_spatial_1p_bf16_512.yaml`: one-prompt smoke eval.
- `eval_official_seethrough3d_spatial_20p_bf16_512.yaml`: short benchmark sanity check.
- `eval_official_seethrough3d_spatial_full_bf16_8bit_512.yaml`: full spatial benchmark run.

## Examples

Train the active 3D_SLN-aligned triple-GNN CVAE:

```bash
python3 -m training.precompute_graph_label_cache \
  --config configs/flux/gnn_pretrain_3dbox_triple_cvae_3dsln.yaml

python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_triple_cvae_3dsln.yaml
```

Train deterministic CLIP/T5 comparison models:

```bash
python3 -m training.precompute_graph_label_cache \
  --config configs/flux/gnn_pretrain_3dbox_deterministic_clip.yaml

python3 -m training.precompute_graph_label_cache \
  --config configs/flux/gnn_pretrain_3dbox_deterministic_t5.yaml

python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_deterministic_clip.yaml

python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_deterministic_t5.yaml
```

Run the one-prompt SeeThrough3D smoke benchmark:

```bash
CONFIG_FILE=configs/flux/eval_official_seethrough3d_spatial_1p_bf16_512.yaml \
bash scripts/benchmark/run_flux_relation_t2icompbench.sh
```

On the 24GB tomago GPUs, the working FLUX inference setting is currently
`bf16` plus `8bit` transformer quantization with `512` image and OSCR sizes.
