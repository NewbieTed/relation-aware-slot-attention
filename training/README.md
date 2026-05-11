# Training Workflow

This FLUX branch now trains only the relation-aware GNN/layout module. FLUX
fine-tuning experiments were removed from the active workflow after the official
SeeThrough3D LoRA plus our GNN-generated OSCR conditions gave the first clear
spatial benchmark gain.

`training.pretrain_graph_encoder` pretrains the graph encoder on SCOP-Depth
relation, position, and 3D box-size targets. The resulting checkpoint is used at
inference time to predict object cuboids, which are rendered into OSCR
conditions for the released SeeThrough3D FLUX LoRA.

The graph encoder supports two layout heads:

- `layout_mode: deterministic`: the original GNN head predicts one center and
  3D size per object.
- `layout_mode: cvae`: a scene-level conditional VAE predicts a distribution
  over plausible 3D layouts. During training it uses
  `q(z | graph, ground_truth_boxes)` and at inference it uses
  `p(z | graph)`.
- `layout_mode: triple_cvae`: a 3D_SLN-style variant that updates contextual
  subject-relation-object edge states, uses separate prior/posterior triple-GNNs,
  samples both scene-level and object-level latents, and runs a decoder
  triple-GNN before predicting centers and 3D sizes.

The graph trainer creates deterministic train/eval/test splits, saves
`data_split.json`, writes losses to `metrics.jsonl` and `metrics.csv`, and saves
reusable checkpoints under the run output directory.

Example graph pretraining:

```bash
./scripts/train/run_graph_pretrain.sh
```

Example CVAE graph pretraining:

```bash
python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_cvae.yaml
```

Example triple-GNN CVAE pretraining:

```bash
python3 -m training.pretrain_graph_encoder \
  --config configs/flux/gnn_pretrain_3dbox_triple_cvae.yaml
```
