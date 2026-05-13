# Training Scripts

This directory contains launch wrappers for model-training jobs. In the current
FLUX branch, only GNN/layout training is active. FLUX fine-tuning is intentionally
not part of the active workflow.

## Scripts

- `run_graph_pretrain.sh`: trains the relation-aware graph encoder on SCOP-Depth
  position, relation, and 3D box-size targets.

## Example

```bash
cd /local1/cse_481_m_l/relation-aware-slot-attention

nohup bash -lc '
source .venv-flux/bin/activate
export CUDA_VISIBLE_DEVICES=7
CONFIG_FILE=configs/flux/gnn_pretrain_3dbox_triple_cvae_3dsln.yaml \
bash scripts/train/run_graph_pretrain.sh
' > logs/gnn_pretrain_3dbox_triple_cvae_3dsln.log 2>&1 &
```

The resulting checkpoint is later passed to FLUX/SeeThrough3D inference with
`--graph-encoder-path`.
