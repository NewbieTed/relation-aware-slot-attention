# FLUX Configs

This directory contains configs for the current FLUX branch. The active path is:
train/debug the GNN, render GNN-predicted 3D cuboids into OSCR conditions, then
run inference with the released SeeThrough3D FLUX LoRA.

## Files

- `gnn_pretrain_3dbox_triple_cvae_3dsln.yaml`: active GNN training config. It
  uses FLUX T5 object-label embeddings, 5 graph layers, L1 min/max box
  reconstruction, scene/object CVAE latents, KL weight `0.1`, and a persistent
  label-embedding cache under `outputs/cache`.
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

Run the one-prompt SeeThrough3D smoke benchmark:

```bash
CONFIG_FILE=configs/flux/eval_official_seethrough3d_spatial_1p_bf16_512.yaml \
bash scripts/benchmark/run_flux_relation_t2icompbench.sh
```

On the 24GB tomago GPUs, the working FLUX inference setting is currently
`bf16` plus `8bit` transformer quantization with `512` image and OSCR sizes.
