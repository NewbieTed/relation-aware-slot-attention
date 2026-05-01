# Training Workflows

This FLUX branch keeps only the current relation-aware training path:

1. `training.pretrain_graph_encoder`
   Pretrains the graph encoder on SCOP-Depth relation, position, and 3D box-size targets.
2. `training.train_relation_flux_lora`
   Freezes FLUX.1-dev, the text encoders, VAE, and the GNN, then trains SeeThrough3D-style LoRA attention processors from GNN-rendered OSCR condition latents.

Both stages create deterministic train/eval/test splits, save `data_split.json`,
write losses to `metrics.jsonl` and `metrics.csv`, and save reusable checkpoints
under the run output directory.

Example graph pretraining:

```bash
./scripts/train/run_graph_pretrain.sh
```

Example FLUX LoRA training:

```bash
./scripts/train/run_flux_lora.sh
```
