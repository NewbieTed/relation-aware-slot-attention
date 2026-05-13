# Training Workflow

This FLUX branch now trains only the relation-aware GNN/layout module. FLUX
fine-tuning experiments were removed from the active workflow after the official
SeeThrough3D LoRA plus our GNN-generated OSCR conditions gave the first clear
spatial benchmark gain.

`training.pretrain_graph_encoder` pretrains the graph encoder on SCOP-Depth
relation, position, and 3D box-size targets. The resulting checkpoint is used at
inference time to predict object cuboids, which are rendered into OSCR
conditions for the released SeeThrough3D FLUX LoRA.

The graph trainer creates deterministic train/eval/test splits, saves
`data_split.json`, writes losses to `metrics.jsonl` and `metrics.csv`, and saves
reusable checkpoints under the run output directory.

Example graph pretraining:

```bash
./scripts/train/run_graph_pretrain.sh
```
