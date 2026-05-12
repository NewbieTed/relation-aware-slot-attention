# FLUX Configs

This directory contains configs for the current FLUX branch. The active path is:
train/debug the GNN, render GNN-predicted 3D cuboids into OSCR conditions, then
run inference with the released SeeThrough3D FLUX LoRA.

## Files

- `gnn_pretrain_3dbox.yaml`: trains the GNN on SCOP-Depth 3D layout targets.
- `gnn_pretrain_3dbox_cvae.yaml`: trains the probabilistic CVAE layout-head
  variant of the GNN.
- `gnn_pretrain_3dbox_triple_cvae.yaml`: trains the 3D_SLN-style triple-GNN
  CVAE with contextual edge states plus scene-level and object-level latents.
- `gnn_pretrain_3dbox_triple_cvae_3dsln.yaml`: longer 3D_SLN-aligned run with
  FLUX T5 object-label embeddings, 5 graph layers, L1 min/max box
  reconstruction, and KL weight `0.1`.
- `eval_official_seethrough3d_spatial_1p_bf16_512.yaml`: one-prompt smoke eval.
- `eval_official_seethrough3d_spatial_20p_bf16_512.yaml`: short benchmark sanity check.
- `eval_official_seethrough3d_spatial_full_bf16_8bit_512.yaml`: full spatial benchmark run.

## Examples

Train the GNN:

```bash
python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox.yaml
```

Train the CVAE GNN variant:

```bash
python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_cvae.yaml
```

Train the triple-GNN CVAE variant:

```bash
python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_triple_cvae.yaml
```

Train the 3D_SLN-aligned triple-GNN CVAE variant:

```bash
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
