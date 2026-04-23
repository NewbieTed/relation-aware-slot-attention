# Training Workflows

This repo now supports a two-stage relation-aware training workflow:

1. `scripts/train/run_graph_pretrain.sh`
   Pretrains only the graph encoder on SCOP-Depth structure supervision. This stage
   learns slot geometry and relation-aware message passing without paying the cost of
   full diffusion training.
2. `scripts/train/run_relation_aware_sd15.sh`
   Runs the full relation-aware Stable Diffusion training loop with LoRA, graph
   conditioning, and modified cross-attention. Pass `INIT_GRAPH_ENCODER=/path/to/graph_encoder.pt`
   to warm-start from stage 1.

The intended schedule is:

- run graph pretraining for many epochs / many steps
- select the best `graph_encoder.pt`
- warm-start the full trainer
- train LoRA + graph encoder + relation-attention for a shorter schedule

This split is useful because the graph encoder typically benefits from much longer
optimization than the LoRA stage.

Both training entrypoints now create deterministic train/eval/test splits, save the
split manifest to `data_split.json`, write loss curves to `metrics.jsonl` and
`metrics.csv`, and report a held-out test loss at the end of training.
