# Prompt Graph

This subpackage converts SCOP-Depth metadata rows into the scene-graph structure
used by the relation-aware GNN.

## Files

- `schema.py`: lightweight dataclasses/types for prompt graph nodes and edges.
- `adapter.py`: converts SCOP-Depth rows into graph payloads with deterministic
  object ordering.

## Usage

Most callers go through `training.prompts.scene_graph_payload_from_row`, which
uses this adapter internally:

```python
from training.prompts import scene_graph_payload_from_row

payload = scene_graph_payload_from_row(metadata_row)
```

The resulting graph is what the GNN sees during pretraining and inference.
