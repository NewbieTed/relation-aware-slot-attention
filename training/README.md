# Training Workflow

This FLUX branch now trains only the relation-aware GNN/layout module. FLUX
fine-tuning experiments were removed from the active workflow after the official
SeeThrough3D LoRA plus our GNN-generated OSCR conditions gave the first clear
spatial benchmark gain.

`training.pretrain_graph_encoder` pretrains the graph encoder on SCOP-Depth
normalized 3D min/max box targets. The resulting checkpoint is used at inference
time to predict object cuboids, which are rendered into OSCR conditions for the
released SeeThrough3D FLUX LoRA.

The active graph encoder uses `layout_mode: triple_cvae`, a 3D_SLN-style variant
that updates contextual subject-relation-object edge states, samples both
scene-level and object-level latents, and runs a decoder triple-GNN before
predicting normalized 3D min/max boxes.

The graph trainer creates deterministic train/eval/test splits, saves
`data_split.json`, writes losses to `metrics.jsonl` and `metrics.csv`, and saves
reusable checkpoints under the run output directory.

Before training, `training.precompute_graph_label_cache` can fill the
`label_embedding_cache` path from the same YAML config. This keeps the frozen
T5/CLIP object-label embeddings out of the hot training loop; training will
still lazily encode and append any unseen labels if the cache is incomplete.

Example graph pretraining:

```bash
python3 -m training.precompute_graph_label_cache \
  --config configs/flux/gnn_pretrain_3dbox_triple_cvae_3dsln.yaml

python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_triple_cvae_3dsln.yaml
```
